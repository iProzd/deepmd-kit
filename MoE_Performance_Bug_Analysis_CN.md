# MoE 实现性能问题分析（中文说明）

## 问题描述

在单卡训练场景下，新 MoE 实现（`deepmd-kit-moe`）比旧实现（`old_moe_deepmdkit`）慢约 **14 倍**：

| 实现 | 100步耗时 | 每步平均 |
|------|----------|---------|
| 旧 MoE | 20.7 s | 0.108 s/batch |
| 新 MoE（修复前） | 144.2 s | **1.505 s/batch** |
| 新 MoE（修复后） | 17.7 s | 0.203 s/batch |

---

## 根本原因

### 关键配置

测试配置中：
- `use_dynamic_sel: True`（动态邻居选择模式）
- `n_experts: 4`，`moe_top_k: 4`
- `use_edge_moe: True`，`use_angle_moe: True`

当 `use_dynamic_sel=True` 时，边的 embedding 不是 `[nb, nloc, nnei, dim]` 这种 batched 格式，而是被压缩成 **flat 格式** `[n_flat, dim]`，其中 `n_flat` 是实际存在的边数（去掉 padding）。

flat 格式的 MoE 走的是 `_moe_flat` 路径（传入非空的 `edge_index`）。

### 原始 `_moe_flat` 的实现（有 Bug）

```python
def _moe_flat(self, x, weights, indices):
    # x:       [n_flat, dim]
    # weights: [n_flat, top_k]
    # indices: [n_flat, top_k]  — 每个 token 选中的 expert id

    output = torch.zeros(n_flat, self.num_out, ...)

    for k in range(self.routed_top_k):      # 循环 top_k=4 次
        w_k   = weights[:, k]               # [n_flat]
        idx_k = indices[:, k]               # [n_flat]
        for e_idx, expert in enumerate(self.experts):  # 循环 n_experts=4 次
            mask = idx_k == e_idx           # bool mask，[n_flat]
            if mask.any():
                expert_out = expert(x[mask])           # ← 问题所在
                output[mask] = output[mask] + w_k[mask].unsqueeze(-1) * expert_out
```

**总循环次数：`top_k × n_experts = 4 × 4 = 16` 次**

### 为什么慢？

#### 原因 1：大量小批量 GPU kernel launch

每次 `expert(x[mask])` 会触发一次 GPU matmul kernel。
- 每次只处理约 `n_flat / n_experts ≈ 25%` 的 token
- GPU 的 CUDA core 利用率在小 batch 时极低
- 本可以一次大矩阵乘完成的计算，被拆成了 16 次小矩阵乘

例如：假设 `n_flat = 10000`（约10k条边），每次 kernel 只处理约 2500 个 token。
GPU 在处理 2500 vs 10000 时，吞吐量（FLOPS/s）差异显著，因为 GPU 擅长大规模并行而非碎片化的小任务。

#### 原因 2：不连续内存访问（bool mask 索引）

`x[mask]` 使用 boolean mask 索引，会产生 **gather** 操作——从 `x` 中挑选满足条件的行，这些行在内存中**不连续**。

不连续内存 → 缓存命中率低 → 内存带宽浪费

相比之下，旧实现先用 `argsort` 对 token 排序，使相同 expert 的 token 在内存中连续，然后用 `index_select`（比 bool mask 更高效）取出，最后做一次**连续内存的大矩阵乘**。

#### 原因 3：scatter 写回开销

```python
output[mask] = output[mask] + ...
```
每次既要从 `output` 中读（gather），又要写回（scatter），16次循环就是 16 次额外的 gather+scatter。

### 对比：旧实现的 `experts_regroup_dynamic`

```python
def experts_regroup_dynamic(xx, topk_index, matrix):
    # 1. 展开路由：每条边对应 topk 个 expert assignment
    token_idx  = ...  # [n_flat * topk]  原始 token 下标
    expert_idx = topk_index.reshape(-1)  # [n_flat * topk]

    # 2. 按 expert 排序 → 相同 expert 的 token 在内存中连续
    order = torch.argsort(expert_idx)
    token_idx  = token_idx[order]
    expert_idx = expert_idx[order]

    # 3. 每个 expert 做一次大的密集矩阵乘
    for e in range(n_experts):           # 只循环 n_experts=4 次
        tok_e = token_idx[start:end]
        x_e   = xx.index_select(0, tok_e)  # 连续内存，高效
        y_e   = torch.matmul(x_e, W[e])    # 一次完整 matmul
        out[tok_e, :, slot] = y_e
```

**总循环次数：`n_experts = 4` 次**，每次处理的 token 量约是新实现单次的 `top_k` 倍，且内存连续。

### 当 top_k == n_experts 时的特殊情况

本测试配置 `top_k = n_experts = 4`，即每个 token 选中**所有 4 个 expert**。

- 旧实现：`experts_regroup_dynamic` 中每个 expert 实际上处理**全部 n_flat 个 token**（因为所有 token 都路由给所有 expert），所以 4 次循环 × n_flat tokens/次 = 4 × n_flat 次 FLOP，但每次都是完整的密集矩阵乘
- 新实现（原始）：16 次循环，每次 n_flat/4 个 token，同样是 4 × n_flat 次 FLOP，但是碎片化执行

总计算量相同，但碎片化执行慢 14 倍，这正好说明 GPU 在大 batch vs 小 batch 时的效率差异有多大。

---

## 修复方案

**仿照 `_moe_batched` 的设计**：先对所有 expert 做**全量** forward，然后用 routing weights 加权求和。

```python
def _moe_flat(self, x, weights, indices):
    # x:       [n_flat, dim]
    # weights: [n_flat, top_k]
    # indices: [n_flat, top_k]

    # 1. 每个 expert 对全量输入做 forward → n_experts 次大矩阵乘
    expert_outputs = [expert(x) for expert in self.experts]
    # 每个 expert_outputs[e]: [n_flat, num_out]

    # 2. 用 scatter_add 把 topk routing weights 累加到对应 expert
    per_expert_weight = torch.zeros(n_flat, self.routed_experts, ...)
    per_expert_weight.scatter_add_(1, indices, weights)
    # per_expert_weight: [n_flat, n_experts]

    # 3. 加权求和
    output = sum(per_expert_weight[:, e].unsqueeze(-1) * expert_outputs[e]
                 for e in range(self.routed_experts))
    # output: [n_flat, num_out]
```

**改变：**
- 循环次数：16 次 → **4 次**（只循环 n_experts）
- 每次 matmul 的 batch size：n_flat/4 → **n_flat**（全量）
- 内存访问：不连续 bool mask → **连续的完整 tensor**
- routing weights 累加：16 次 scatter → **1 次 scatter_add_**

---

## 修复后的剩余 2 倍差距

修复后新实现仍比旧实现慢约 2 倍，原因是结构性差异：

### 1. 权重存储方式不同

| 实现 | 存储方式 | matmul 方式 |
|------|---------|-----------|
| 旧 MoE | 3D tensor `[in, out, n_exp]` | `matmul(x, W[e])` — 内存连续，无额外开销 |
| 新 MoE | `ModuleList` of `MLPLayer` | `F.linear(x, matrix.t(), bias)` — 需要转置视图 |

旧实现把所有 expert 的权重放在一个 3D tensor 里，理论上可以做一次批量 bmm；新实现每个 expert 是独立的 `MLPLayer`，需要分 4 次调用。

### 2. 中间 tensor 分配

新实现每次 forward 会分配 4 个 `[n_flat, num_out]` 的中间 tensor（每个 expert 一个），旧实现只需要 1 个输出 tensor。额外的内存分配和 GC 压力会带来一定开销。

### 3. 架构层面的额外计算

新实现额外引入了 `node_self_mlp_moe` 和 `node_sym_linear_moe`（node MoE），这些也需要 MoE forward，增加了计算量（虽然旧实现也有，但用的是更高效的 3D 矩阵）。

### 4. `F.linear` 的 CUBLAS 问题

训练日志中有警告：
```
gemm_and_bias error: CUBLAS_STATUS_NOT_INITIALIZED ... Will attempt to recover by calling unfused cublas path
```
说明某些情况下 fused CUBLAS kernel 初始化失败，回退到更慢的 unfused 路径。

---

## 修改的文件

**`deepmd/pt/model/network/moe.py`** — `_moe_flat` 方法

修改位置：第 245-286 行（将双重循环替换为全量计算+加权求和）

---

## 经验总结

**GPU 性能优化的核心原则：**
1. **大 batch 优于小 batch**：GPU 擅长大规模并行，碎片化的小 kernel launch 会浪费大量时间在调度和同步上
2. **连续内存访问**：bool mask 索引会产生不连续的 gather 操作，降低缓存效率
3. **减少 kernel launch 次数**：每次 Python 层面的循环都可能触发一次 GPU kernel，尽量合并操作
4. **当 top_k = n_experts 时**：所有 expert 都被选中，直接全量计算后加权求和是最优方案；只有 top_k < n_experts（真正稀疏路由）时，按 expert 分组的方式才有意义（可以跳过权重为 0 的 expert）
