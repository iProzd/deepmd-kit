# MoE 实现分析与对比文档

## 目录
1. [项目总体状况](#1-项目总体状况)
2. [旧版 MoE 单卡实现 (old_moe_deepmdkit)](#2-旧版-moe-单卡实现)
3. [新版 MoE 单卡实现 (deepmd-kit-moe)](#3-新版-moe-单卡实现)
4. [新旧单卡实现对比](#4-新旧单卡实现对比)
5. [新版多卡 EP+DP 实现](#5-新版多卡-epdp-实现)
6. [单卡 vs 多卡 EP 对比](#6-单卡-vs-多卡-ep-对比)

---

## 1. 项目总体状况

### 已完成的 Bug 修复

**Bug 1: Expert 参数梯度为零**
- 根因: `moe_ep_ops.py` 中 `_EPMoEBackward.forward()` 使用 `torch.autograd.grad()` 计算输入梯度，但该函数**不会**累积 expert 参数 (weight/bias) 的 `.grad`
- 修复: 在 `torch.autograd.grad()` 之后增加 `result.backward()` 调用
- 文件: `deepmd/pt/model/network/moe_ep_ops.py`

**Bug 2: Expert 梯度缩放不匹配**
- 根因: Expert 梯度通过 A2A backward 累积了所有 EP rank 的贡献（求和），而 shared 参数是取平均
- 修复: 在 `_ep_gradient_sync()` 中对 expert 梯度除以 `ep_size`
- 文件: `deepmd/pt/train/training.py`

### 已完成的速度优化

| 优化项 | 改动 | 效果 |
|--------|------|------|
| Ghost term 计算 | `expert(zeros)` → `sum(p.sum() for p in params) * 0.0` | 消除 8 次无用前向传播 |
| 梯度同步批量化 | 逐参数 all_reduce → flatten + 单次 all_reduce | 通信调用从 O(N) 降到 2 |
| Dispatch 元数据 | 逐 GPU `.item()` 循环 → `torch.bincount` | GPU→CPU 同步从 8 次降到 1 次 |
| Sort-based expert dispatch | clone-per-expert → argsort + 连续切片 + cat | 消除每次 forward/backward 中 8 次全张量拷贝 |
| index_add 聚合 | clone + scatter → 收集后单次 index_add | 消除 GPU-level 路径中的重复 clone |

**总体速度提升: 1.88 s/step → 1.28 s/step (32% 加速)**

---

## 2. 旧版 MoE 单卡实现

代码路径: `/mnt/data_nas/zhangd/claude_space/old_moe_deepmdkit`

### 2.1 整体架构

旧版没有独立的 MoE 模块文件，MoE 逻辑分散在 `mlp.py` 的多个类中:

```
mlp.py 中的 MoE 相关类:
├── MOErouter          — Router (gate)
├── mlp_layer_moe      — 路由 expert 层 (batched 模式)
├── mlp_layer_moe_dynamic_sel — 路由 expert 层 (dynamic_sel 模式)
├── share_mlp_layer_moe — 共享 expert 层
├── MOE_replace_mlp    — 顶层封装，替代普通 MLPLayer
├── MOLEMLPLayer       — MOLE 变体 (einsum-based)
└── experts_regroup_only() — 按 expert 分组计算的辅助函数
```

### 2.2 Router 机制

```python
# MOErouter (mlp.py:1600)
class MOErouter(nn.Module):
    def __init__(self, num_in, num_out, topk):
        self.mlp = MLPLayer(num_in, num_out)  # 线性层，无 bias，无激活

    def forward(self, xx):
        alpha = self.mlp(xx)                          # [nb, nloc, routed_experts]
        topk_alpha, topk_indices = torch.topk(alpha, k=self.topk, dim=-1)
        probs = F.softmax(topk_alpha, dim=-1)         # [nb, nloc, topk]
        return probs, topk_indices
```

**Router 输入**: `atom_type_ebd`，即 `type_embedding(extended_atype)` — 每个原子的 type embedding，shape `[nb, nall, n_dim]`（或 `[nb, nloc, n_dim]`）。当有 `fparam` 时，`atom_type_ebd = cat([type_embedding, fparam_embd], dim=-1)`，维度变为 `2 * n_dim`。

**关键特性**: Router 的输入是 type embedding（按原子类型查表得到），不是特征向量。因此同类型原子的路由权重完全相同，保证了 MD 连续性。

### 2.3 Expert 权重结构

旧版的 expert 不是独立的 `nn.Module`，而是**共享一个三维权重张量**:

```python
# mlp_layer_moe (mlp.py:470)
class mlp_layer_moe(nn.Module):
    def __init__(self, num_in, num_out, numb_experts, ...):
        # 所有 expert 的权重合并为一个 3D 张量
        self.matrix = nn.Parameter(torch.empty(num_in, num_out, numb_experts))
        self.bias = nn.Parameter(torch.empty(num_out))  # 所有 expert 共享 bias
```

注意: **所有 expert 共享同一个 bias**，只有 weight matrix 的第三维区分不同 expert。

### 2.4 Expert 计算流程

旧版使用 `experts_regroup_only()` 函数进行计算:

```python
def experts_regroup_only(xx, topk_index, matrix):
    """
    xx:         [nb, nloc, *dims_mid, num_in]
    topk_index: [nb, nloc, topk]
    matrix:     [num_in, num_out, n_experts]
    return:     [nb, nloc, *dims_mid, num_out, topk]
    """
    B = nb * nloc
    x = xx.reshape(B, S, num_in)           # S = prod(dims_mid)
    W = matrix.permute(2, 0, 1)            # [n_experts, num_in, num_out]

    # 按 expert 排序所有 (token, slot) 对
    expert_idx = topk_index.reshape(-1)     # [B * topk]
    order = torch.argsort(expert_idx)
    counts = torch.bincount(expert_idx, minlength=n_experts)

    # 逐 expert 做 matmul
    out = torch.empty(B, S, num_out, topk)
    for e in range(n_experts):
        tok_e = token_idx[start:end]        # 属于 expert e 的 token 索引
        slot_e = slot_idx[start:end]        # 对应的 topk slot
        x_e = x.index_select(0, tok_e)      # [Ne, S, num_in]
        y_e = torch.matmul(x_e, W[e])       # [Ne, S, num_out]
        out[tok_e, :, :, slot_e] = y_e      # 写回对应位置
```

**输出 shape**: `[nb, nloc, *dims_mid, num_out, topk]` — 最后一维是 topk 维度。

### 2.5 加权合并

在 `MOE_replace_mlp.forward()` 中完成:

```python
def forward(self, xx, atom_type_ebd, nei_index=None):
    alpha, topk_indices = self.router(atom_type_ebd)  # alpha: [nb, nloc, topk]
    yy = self.moe_layer(xx, topk_indices)             # [nb, nloc, *mid, num_out, topk]

    # 加权合并: 沿 topk 维度做矩阵乘法
    yy_extend = torch.matmul(
        yy.view(nb*nloc, -1, topk),    # [..., topk]
        alpha.view(nb*nloc, topk, 1)   # [topk, 1]
    )                                   # [..., 1] → squeeze → [nb, nloc, *mid, num_out]

    # 加上 shared expert
    if self.share_expert > 0:
        yy_extend += self.share_layer(xx)
    return yy_extend
```

### 2.6 Shared Expert

`share_mlp_layer_moe` 也使用三维权重张量 `[num_in, num_out, numb_experts]`，但 forward 中直接对所有 expert 维度求和:

```python
def forward(self, xx):
    yy_tmp = torch.matmul(xx.view(-1, num_in), matrix.reshape(num_in, numb_experts * num_out))
    yy = yy_tmp.view(*dims_mid, num_out, numb_experts) + bias
    yy = activate(yy)
    yy = torch.sum(yy, dim=-1)  # 对所有 shared expert 求和
```

### 2.7 在 RepFlowLayer 中的集成

旧版通过 `MOE_replace_mlp` 直接替换 `MLPLayer`:

```python
# repflow_layer.py
if self.n_experts > 1:
    self.node_self_mlp = MOE_replace_mlp(n_dim, n_dim, self.moe_dim, ...)
else:
    self.node_self_mlp = MLPLayer(n_dim, n_dim, ...)

# forward 中:
if self.n_experts > 1:
    node_self_mlp = self.node_self_mlp(node_ebd, atom_type_ebd)
else:
    node_self_mlp = self.act(self.node_self_mlp(node_ebd))
```

MoE 应用于 7 个位置: `node_self_mlp`, `node_sym_linear`, `node_edge_linear`, `edge_self_linear`, `edge_angle_linear1`, `edge_angle_linear2`, `angle_self_linear`。

---

## 3. 新版 MoE 单卡实现

代码路径: `/mnt/data_nas/zhangd/claude_space/deepmd-kit-moe`

### 3.1 整体架构

新版将 MoE 逻辑集中到独立模块:

```
deepmd/pt/model/network/
├── moe.py          — MoELayer: router + experts + 单卡/EP forward
├── moe_ep_ops.py   — EP 专用 autograd Function (A2A + double-backward)
└── moe_fused.py    — FusedMoELayer: 融合多个 MoE 减少 A2A 轮次
```

### 3.2 Router 机制

```python
# MoELayer._route() (moe.py:158-189)
def _route(self, type_embeddings, atom_types):
    # type_embeddings: [ntypes, tebd_dim]  — 注意: 是 ntypes 行，不是 nall 行
    logits = self.gate(type_embeddings)              # [ntypes, routed_experts]
    topk_logits, topk_indices = torch.topk(logits, k=self.routed_top_k, dim=-1)
    weights = F.softmax(topk_logits, dim=-1)         # [ntypes, routed_top_k]
    atom_weights = weights[atom_types]               # [nb, nloc, routed_top_k]
    atom_indices = topk_indices[atom_types]           # [nb, nloc, routed_top_k]
    return atom_weights, atom_indices
```

**Router 输入**: `type_embeddings = type_embedding(arange(ntypes))`，shape `[ntypes, tebd_dim]`。Gate 只计算 `ntypes` 行（如 89 种元素类型），然后通过 `atom_types` 索引扩展到每个原子。

### 3.3 Expert 权重结构

新版每个 expert 是**独立的 `MLPLayer`** (nn.Module):

```python
# MoELayer.__init__() (moe.py:127-139)
self.experts = nn.ModuleList([
    MLPLayer(num_in, num_out, bias=bias, activation_function=activation_function, ...)
    for i in range(self.experts_per_gpu)
])
```

每个 expert 有独立的 `weight: [num_in, num_out]` 和 `bias: [num_out]`。

### 3.4 Expert 计算流程 (单卡)

**Batched 模式** (`_moe_batched`, moe.py:288-335):

```python
def _moe_batched(self, x, atom_weights, atom_indices):
    # x: [nb, nloc, *mid, dim_in]

    # Step 1: 所有 expert 处理所有 token
    expert_outputs = [expert(x) for expert in self.experts]

    # Step 2: 构建 per-expert 权重矩阵
    per_expert_weight = zeros(nb, nloc, routed_experts)
    for k in range(routed_top_k):
        per_expert_weight.scatter_add_(2, atom_indices[:,:,k], atom_weights[:,:,k])

    # Step 3: 加权求和
    output = sum(per_expert_weight[:,:,e] * expert_outputs[e] for e in range(n_experts))

    # Step 4: 加上 shared expert
    for se in self.shared_experts:
        output = output + se(x)
```

**Flat 模式** (`_moe_flat`, moe.py:245-286): 逻辑相同，但输入 shape 为 `[n_flat, dim]`。

**关键特性**: 单卡模式下，**每个 expert 处理所有 token**。路由权重只控制线性组合的系数。这是因为 `n_experts` 通常较小 (3-8)，全量计算比动态索引更高效。

### 3.5 Shared Expert

新版的 shared expert 也是独立的 `MLPLayer`:

```python
self.shared_experts = nn.ModuleList([
    MLPLayer(num_in, num_out, ...)
    for i in range(self.share_expert)
])
```

Forward 中直接加到输出上:
```python
for se in self.shared_experts:
    output = output + se(x)
```

### 3.6 在 RepFlowLayer 中的集成

新版通过统一的 `MoELayer` 接口:

```python
# repflow_layer.py
if use_node_moe:
    self.node_self_mlp = MoELayer(n_dim, n_dim, n_experts, top_k, tebd_dim, ...)
else:
    self.node_self_mlp = MLPLayer(n_dim, n_dim, ...)

# forward 中:
if self.node_self_mlp_is_moe:
    node_self_mlp = self.node_self_mlp(node_ebd, type_embeddings, atom_types)
else:
    node_self_mlp = self.act(self.node_self_mlp(node_ebd))
```

MoE 可选应用于 7 个位置，通过 `use_node_moe`, `use_edge_moe`, `use_angle_moe` 三个开关控制。

---

## 4. 新旧单卡实现对比

### 4.1 核心差异总表

| 维度 | 旧版 (old_moe_deepmdkit) | 新版 (deepmd-kit-moe) |
|------|--------------------------|----------------------|
| **代码组织** | MoE 逻辑分散在 `mlp.py` 的 6+ 个类中 | 独立 `moe.py` + `moe_ep_ops.py` + `moe_fused.py` |
| **Router 输入** | `atom_type_ebd`: 每个原子的 type embedding `[nb, nall, n_dim]` | `type_embeddings`: 每种类型的 embedding `[ntypes, tebd_dim]`，再按 `atom_types` 索引 |
| **Router 计算量** | Gate 对 `nb * nall` 个向量做线性变换 | Gate 只对 `ntypes` 个向量做线性变换（通常 ntypes << nall） |
| **Expert 权重** | 共享 3D 张量 `[num_in, num_out, n_experts]` | 独立 `nn.Module`，每个 expert 有自己的 `[num_in, num_out]` weight 和 `[num_out]` bias |
| **Expert bias** | 所有 routed expert 共享一个 bias | 每个 expert 有独立 bias |
| **Expert 计算** | `experts_regroup_only()`: 按 expert 排序 → 逐 expert `matmul(x_e, W[e])` → 写回 | 所有 expert 处理所有 token → `scatter_add` 构建权重 → 加权求和 |
| **输出中间形态** | `[nb, nloc, *mid, num_out, topk]` — 保留 topk 维度 | 直接输出 `[nb, nloc, *mid, num_out]` — 已加权合并 |
| **加权合并** | 最后通过 `matmul(yy, alpha)` 沿 topk 维度合并 | 在 expert 循环中通过 `per_expert_weight * expert_output` 逐 expert 累加 |
| **Shared expert** | `share_mlp_layer_moe`: 3D 权重 + `sum(dim=-1)` | 独立 `MLPLayer` 列表，逐个加到输出 |
| **多卡支持** | 无 | 完整 EP + DP 支持 |
| **MoE 融合** | 无 | `FusedMoELayer` 可融合同输入的 MoE 层 |
| **Double-backward** | 无特殊处理 | `moe_ep_ops.py` 中两层嵌套 autograd Function 防止 NCCL 死锁 |

### 4.2 Router 差异详解

**旧版**:
```
DPA3.forward():
    atom_type_ebd = type_embedding(extended_atype)  # [nb, nall, n_dim]
    → 传入 RepFlowLayer → 传入 MOE_replace_mlp.forward(xx, atom_type_ebd)
    → MOErouter(atom_type_ebd)
    → gate(atom_type_ebd)  # 对 nb*nall 个向量做线性变换
```

**新版**:
```
DPA3.forward():
    type_embeddings = type_embedding(arange(ntypes))  # [ntypes, tebd_dim]
    → 传入 RepFlowLayer → 传入 MoELayer.forward(x, type_embeddings, atom_types)
    → _route(type_embeddings, atom_types)
    → gate(type_embeddings)  # 只对 ntypes 个向量做线性变换
    → weights[atom_types]    # 按原子类型索引扩展
```

新版的 router 计算量更小: gate 只需处理 `ntypes` 行（如 89），而旧版需处理 `nb * nall` 行（如 1 * 192 = 192）。虽然差异不大，但新版的设计更清晰地表达了"同类型原子路由相同"这一语义。

### 4.3 Expert 权重结构差异

**旧版 — 共享 3D 张量**:
```python
self.matrix = nn.Parameter(torch.empty(num_in, num_out, n_experts))
# 所有 expert 的权重紧凑存储在一个张量中
# 计算: W = matrix.permute(2, 0, 1)  → [n_experts, num_in, num_out]
#        y_e = matmul(x_e, W[e])
```

优点: 内存连续，可以用单次 matmul 处理（如 `einsum`）
缺点: 所有 expert 共享 bias；不易拆分到多 GPU

**新版 — 独立 nn.Module**:
```python
self.experts = nn.ModuleList([MLPLayer(...) for i in range(n_experts)])
# 每个 expert 是独立的 nn.Module，有自己的 weight 和 bias
# 计算: output_e = expert_e(x)  # 标准 F.linear
```

优点: 每个 expert 有独立 bias；天然支持 EP 分片（每个 GPU 只创建 `experts_per_gpu` 个）
缺点: 单卡模式下需要逐 expert 循环调用

### 4.4 计算策略差异

**旧版 — "按 expert 分组 matmul"**:
1. 将所有 `(token, topk_slot)` 对按 expert_id 排序
2. 对每个 expert，取出属于它的 token 子集
3. 用该 expert 的权重切片 `W[e]` 做 matmul
4. 将结果写回对应位置
5. 最终输出保留 topk 维度: `[nb, nloc, *mid, num_out, topk]`
6. 在外层通过 `matmul(yy, alpha)` 合并

**新版 — "全量计算 + 加权求和"**:
1. 每个 expert 处理所有 token（不做子集选择）
2. 通过 `scatter_add` 构建 `[nb, nloc, routed_experts]` 的权重矩阵
3. 逐 expert 做 `weight * output` 累加
4. 直接输出已合并的结果: `[nb, nloc, *mid, num_out]`

新版的策略在 `n_experts` 较小时更高效（避免动态索引的开销），但在 `n_experts` 很大时会有冗余计算（每个 expert 都处理了不属于它的 token）。对于当前配置 (`experts_per_gpu=8`)，两种策略的效率差异不大。

---

## 5. 新版多卡 EP+DP 实现

### 5.1 设备网格

```python
# training.py: _setup_expert_parallelism()
# 例: world_size=8, ep_size=8, dp_size=1
mesh = init_device_mesh("cuda", (dp_size, ep_size), mesh_dim_names=("dp", "ep"))
ep_group = mesh["ep"].get_group()  # EP 通信组
dp_group = mesh["dp"].get_group()  # DP 通信组
```

当前配置: `world_size=8, ep_size=8, dp_size=1`（纯 EP，无数据并行）。

### 5.2 Expert 分片

每个 GPU 只创建 `routed_experts // ep_size` 个 expert:

```
总共 64 个 routed expert + 1 个 shared expert
GPU 0: experts[0..7]   + shared_expert (复制)
GPU 1: experts[8..15]  + shared_expert (复制)
...
GPU 7: experts[56..63] + shared_expert (复制)
```

### 5.3 EP Forward 流程 (Expert-Level)

`_moe_ep_flat_expert_level()` (moe.py:624-682):

```
输入: x [n_flat, dim], weights [n_flat, topk], indices [n_flat, topk]

Step 1: Token 展开
    x_topk = x.repeat_interleave(topk)     # [n_flat * topk, dim]
    expert_ids = indices.reshape(-1)         # [n_flat * topk] — 全局 expert ID

Step 2: Dispatch 元数据 (_ep_dispatch_info)
    target_gpu = expert_id // experts_per_gpu
    send_perm = argsort(target_gpu)          # 按目标 GPU 排序
    send_splits = bincount(target_gpu)       # 每个 GPU 发送多少 token
    recv_splits ← all_to_all(send_splits)    # 交换接收计数
    recv_eids ← all_to_all(local_expert_ids) # 交换本地 expert ID

Step 3: Fused A2A + Expert Compute + A2A (moe_ep_ops.py)
    ┌─ Dispatch A2A: 将 token 发送到 expert 所在 GPU
    │   h_recv = a2a(x_sorted, send_splits, recv_splits)
    │
    ├─ Local Expert Compute:
    │   sort by expert_id → contiguous slicing
    │   for each local expert:
    │       result = expert(h_sorted[offset:offset+cnt])
    │
    └─ Combine A2A: 将结果发回原 GPU
        h_ret = a2a(expert_out, recv_splits, send_splits)

Step 4: Un-permute + 加权合并
    h_unperm[send_perm] = h_ret
    h_topk = h_unperm.view(n_flat, topk, num_out)
    output = (h_topk * weights.unsqueeze(-1)).sum(dim=1)

Step 5: Ghost terms (autograd 图连通性)
    ghost_sum = sum(p.sum() for all expert params) * 0.0
    output = output + ghost_sum

Step 6: Shared expert
    for se in shared_experts:
        output = output + se(x)
```

### 5.4 EP Forward 流程 (GPU-Level)

`_moe_ep_flat_gpu_level()` (moe.py:684-732):

GPU-level 是 expert-level 的优化版本。当 `topk > 1` 且同一 GPU 上有多个被选中的 expert 时，expert-level 会发送重复的 token 副本。GPU-level 通过去重减少通信量:

```
Expert-level: 每个 (token, expert) 对发送一份 → 通信量 = n_flat * topk * dim
GPU-level:    每个 (token, target_gpu) 对只发送一份 → 通信量 = n_dedup * dim

例: token A 选中了 expert 3 和 expert 5，都在 GPU 0 上
    Expert-level: 发送 2 份 token A 到 GPU 0
    GPU-level:    只发送 1 份 token A 到 GPU 0，附带 eid_matrix=[3,5] 和 weight_matrix=[w3,w5]
```

接收端在本地对多个 expert 的输出做加权求和后再发回，进一步减少 combine A2A 的通信量。

### 5.5 Double-Backward 安全机制

ML 势函数需要二阶导数: energy → force (1st) → virial/hessian (2nd)。

**问题**: 在 `loss.backward()` (2nd backward) 中，PyTorch autograd 的拓扑排序可能在不同 rank 上产生不同的 backward 调用顺序。如果这些 backward 包含 NCCL All-to-All，就会因为顺序不匹配导致死锁。

**解决方案**: 两层嵌套的 autograd Function:

```
_EPMoEForward.forward()     — 包含 A2A (确定性顺序)
_EPMoEForward.backward()    — 调用 _EPMoEBackward.apply()
_EPMoEBackward.forward()    — 包含 A2A (确定性顺序，= 1st backward)
_EPMoEBackward.backward()   — 无 A2A，返回全 None (= 2nd backward)
```

所有 A2A 通信都在 `forward()` 方法中执行（确定性顺序），2nd backward 中完全没有跨 rank 通信，因此 autograd 可以以任意顺序处理这些节点。

代价: 2nd-order force 对 expert 参数梯度的贡献被丢弃（返回 None），但主要的 1st-order energy 梯度 (dE/dθ) 通过 forward graph 正确流动。

### 5.6 梯度同步

DDP 无法处理 EP（它只能用一个 process_group），因此 EP 模式跳过 DDP，手动同步梯度:

```python
# training.py: _ep_gradient_sync()

# Shared 参数 (gate, shared_experts, fitting_net, encoder 等):
#   → 全局 all_reduce → 除以 world_size
flat = flatten(shared_grads)
dist.all_reduce(flat)           # 全局组
flat /= world_size

# Routed expert 参数:
#   → dp_group 内 all_reduce → 除以 dp_size → 再除以 ep_size
flat = flatten(expert_grads)
dist.all_reduce(flat, group=dp_group)  # DP 组
flat /= dp_size
for g in expert_grads:
    g /= ep_size  # A2A backward 累积的是 sum，需要转为 average
```

### 5.7 FusedMoELayer

当 `fuse_moe_mlps=True` 时，共享相同输入的 MoE 层被融合:

```
Fusion A: node_edge_linear + edge_self_linear → edge_fused_moe
          (两者都以 edge_info 为输入)
Fusion B: edge_angle_linear1 + angle_self_linear → angle_fused_moe
          (两者都以 angle_info 为输入)
```

融合后创建一个 `num_out = sum(output_dims)` 的宽 MoE，只需一轮 A2A，然后 split 输出。将 A2A 轮次从 2N 减少到 2。

---

## 6. 单卡 vs 多卡 EP 对比

| 维度 | 单卡模式 | 多卡 EP 模式 |
|------|----------|-------------|
| **Expert 数量** | 所有 `routed_experts` 个 expert 在一个 GPU 上 | 每个 GPU 只有 `routed_experts / ep_size` 个 |
| **计算策略** | 每个 expert 处理所有 token（全量计算） | 每个 expert 只处理路由到它的 token（稀疏计算） |
| **通信** | 无 | 2 轮 All-to-All (dispatch + combine) per MoE layer |
| **Token 路由** | 路由权重只影响线性组合系数 | 路由权重决定 token 物理发送到哪个 GPU |
| **Shared expert** | 直接加到输出 | 每个 GPU 上复制一份，直接加到输出 |
| **梯度同步** | DDP 自动处理 | 手动: shared 全局 all_reduce + expert dp_group all_reduce + /ep_size |
| **Ghost terms** | 不需要 | 需要: 确保所有 rank 的 autograd 图结构一致 |
| **Double-backward** | 标准 autograd | 两层嵌套 Function 防止 NCCL 死锁 |
| **内存** | 所有 expert 权重在一个 GPU 上 | expert 权重分布在多个 GPU 上，单卡内存更小 |
| **扩展性** | 受单卡内存限制 | 可扩展到更多 expert (如 64 个 routed expert) |

### 6.1 计算效率对比

**单卡**: 每个 expert 处理 `n_flat` 个 token，共 `n_experts` 个 expert
- 总计算量: `n_experts × n_flat × num_in × num_out`
- 冗余计算: 每个 token 被所有 expert 处理，但只有 topk 个 expert 的输出有非零权重

**多卡 EP**: 每个 GPU 上的 expert 只处理路由到它的 token
- 每 GPU 计算量: `experts_per_gpu × (n_flat × topk / ep_size) × num_in × num_out`（近似均匀分布时）
- 无冗余计算，但增加了 A2A 通信开销

### 6.2 为什么需要 Ghost Terms

单卡模式下，所有 expert 都参与前向计算，autograd 图自然包含所有参数。

EP 模式下，某些 GPU 上的某些 expert 可能没有收到任何 token（负载不均衡时）。如果这些 expert 不在 autograd 图中:
1. `loss.backward()` 不会遍历到这些 expert 的参数
2. 不同 rank 的 autograd 图结构不同
3. 2nd backward 中拓扑排序不一致 → NCCL 死锁

Ghost terms 通过 `sum(p.sum() for unused expert params) * 0.0` 将未使用 expert 的参数以零贡献加入图中，确保所有 rank 的图结构一致。
