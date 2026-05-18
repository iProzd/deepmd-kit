# SeZM MoE EP+DP 工程规范

> 本文档是 SeZM MoE 并行实现的**完整工程规范**。在写任何代码之前必须读完。每个 Step 的实现必须严格遵循本文档列出的约束、shape、和不变量。配套读物：`CLAUDE.md`（环境与规则）、`skills/sezm-moe-design/SKILL.md`（设计模式速查）。
>
> 当前实现/验证进度不写在本文档中，统一记录在 `PROGRESS.md`，以保持 `SPEC.md` 作为稳定设计契约。

______________________________________________________________________

## 目录

1. [目标与范围](#1-%E7%9B%AE%E6%A0%87%E4%B8%8E%E8%8C%83%E5%9B%B4)
1. [关键术语与符号](#2-%E5%85%B3%E9%94%AE%E6%9C%AF%E8%AF%AD%E4%B8%8E%E7%AC%A6%E5%8F%B7)
1. [支持的配置（不支持的必须 raise）](#3-%E6%94%AF%E6%8C%81%E7%9A%84%E9%85%8D%E7%BD%AE%E4%B8%8D%E6%94%AF%E6%8C%81%E7%9A%84%E5%BF%85%E9%A1%BB-raise)
1. [核心架构](#4-%E6%A0%B8%E5%BF%83%E6%9E%B6%E6%9E%84)
1. [梯度同步推导](#5-%E6%A2%AF%E5%BA%A6%E5%90%8C%E6%AD%A5%E6%8E%A8%E5%AF%BC)
1. [Step-by-Step 实现计划](#6-step-by-step-%E5%AE%9E%E7%8E%B0%E8%AE%A1%E5%88%92)
1. [测试矩阵](#7-%E6%B5%8B%E8%AF%95%E7%9F%A9%E9%98%B5)
1. [配置 schema](#8-%E9%85%8D%E7%BD%AE-schema)
1. [关键不变量与守护](#9-%E5%85%B3%E9%94%AE%E4%B8%8D%E5%8F%98%E9%87%8F%E4%B8%8E%E5%AE%88%E6%8A%A4)
1. [DPA3 参考代码索引](#10-dpa3-%E5%8F%82%E8%80%83%E4%BB%A3%E7%A0%81%E7%B4%A2%E5%BC%95)

______________________________________________________________________

## 1. 目标与范围

### 1.1 目标

把 SeZM 描述符 `SO2Convolution` 内部的 **SO(2) Linear stack**（默认 4 层）替换为 **MoE Expert Parallelism (EP) + Data Parallelism (DP)** 路径。每条边的 `F=n_focus` 个 focus 槽各自由不同的 expert 计算，其中：

- **前 topk 个 slot** 由 router 选中的 routing experts 处理（走 A2A）
- **后 S=n_shared_experts 个 slot** 由 shared experts 处理（不走 A2A，本地算）

最终输出形状保持 `(E, F=topk+S, D_m, Cf)` 进入 SO(2) 卷积的下游（rotate-back / aggregate / FFN）。

### 1.2 范围

- **MoE 化**：SO2Convolution 中 4 层 SO2Linear + 中间激活 + 可能的 bias_correction + 可能的 layer_scale（共享）
- **不 MoE 化**：rotate-to-local / rotate-back / radial_modulation / attention / FFN / `EquivariantRMSNorm` / 顶层 type_embedding / 残差
- **不支持** v1：`torch.compile`、`so2_norm=True`、`use_so2_attn_res != "none"`

### 1.3 性能目标

- 单卡 MoE（`ep_group=None`）相对 no-MoE 同配置 overhead < 15%
- 多卡 EP+DP forward/backward/二阶导 **不死锁**
- 力损失训练正确（二阶导精确）

______________________________________________________________________

## 2. 关键术语与符号

| 符号                | 含义                                                          | 默认值（DPA-4 input_mptrj.json） |
| ------------------- | ------------------------------------------------------------- | -------------------------------- |
| `N`                 | 局部节点数 = nf × nloc                                        | —                                |
| `E`                 | 有效边数（含 +1 dummy）                                       | —                                |
| `D`                 | SO(3) 系数维度 = `(lmax+1)²`                                  | 16                               |
| `D_m`               | reduced m-major 维度 = `(lmax+1) + 2·(lmax-mmax)·mmax + ...`  | 10（lmax=3, mmax=1）             |
| `C`                 | 描述符通道数（channels）                                      | 64                               |
| `Cf`                | 每 focus 通道数（focus_dim 或 channels）                      | 64（focus_dim=0 时）             |
| `F`                 | n_focus，**MoE 下 = topk + S**                                | 1（非 MoE 默认）                 |
| `H`                 | hidden_channels = F × Cf                                      | 64                               |
| **MoE 新增**        |                                                               |                                  |
| `topk`              | router 选中的 routing expert 数                               | 用户配置                         |
| `S`                 | n_shared_experts                                              | 用户配置                         |
| `n_routing_experts` | 总 routing expert 数（≥ topk）                                | 用户配置                         |
| `n_experts_per_gpu` | 每 GPU 持有的 routing expert 数 = n_routing_experts / ep_size | 派生                             |
| `ep_size`           | EP 组大小                                                     | 用户配置                         |
| `dp_size`           | DP 组大小 = world_size / ep_size                              | 派生                             |
| `world_size`        | 总 GPU 数 = ep_size × dp_size                                 | 派生                             |
| `routing_input`     | router 输入选择：`"dst"`/`"src"`/`"src+dst"`                  | `"dst"`                          |

______________________________________________________________________

## 3. 支持的配置（不支持的必须 raise）

凡是下面表格里"必须为 X"那一列与用户配置不符的，**初始化时直接 `raise ValueError(具体哪个不符)`**。**禁止任何 fallback 路径**。

| 配置项                                                    | v1 必须                                       | 校验位置                     |
| --------------------------------------------------------- | --------------------------------------------- | ---------------------------- |
| `use_compile`                                             | `False`                                       | DescrptSeZM `__init__`       |
| `so2_norm`                                                | `False`                                       | SO2Convolution `__init__`    |
| `use_so2_attn_res`                                        | `"none"`                                      | SO2Convolution `__init__`    |
| `n_focus`                                                 | `== topk + n_shared_experts`                  | SO2Convolution `__init__`    |
| `n_routing_experts % ep_size`                             | `== 0`（每 GPU 整数个 expert）                | MoESO2Convolution `__init__` |
| `topk`                                                    | `>= 1` 且 `<= n_routing_experts`              | MoESO2Router `__init__`      |
| `routing_input`                                           | `"dst"` / `"src"` / `"src+dst"`               | MoESO2Router `__init__`      |
| `n_shared_experts`                                        | `>= 0`                                        | MoESO2Convolution `__init__` |
| `world_size % ep_size`                                    | `== 0`                                        | init_ep_dp_groups            |
| `hidden_channels = F × Cf`                                | radial_hidden_proj 必须正确对齐               | SO2Convolution `__init__`    |
| `mlp_bias=True` 且 `routing_input="src+dst"` 且 `topk` 大 | 可以支持，但要确认 bias_correction 通道数正确 | MoESO2Convolution `__init__` |

错误信息要具体，例如：

```
ValueError: SeZM MoE requires `n_focus == topk + n_shared_experts`, got
            n_focus=4, topk=2, n_shared_experts=1 (sum=3). Please adjust your config.
```

______________________________________________________________________

## 4. 核心架构

### 4.1 进程组与 expert 分布（与 DPA3 一致）

```
world_size = ep_size × dp_size

8 GPU 例: ep_size=4, dp_size=2

           EP rank 0   EP rank 1   EP rank 2   EP rank 3
DP rank 0:   GPU 0       GPU 1       GPU 2       GPU 3     ← ep_group_0
DP rank 1:   GPU 4       GPU 5       GPU 6       GPU 7     ← ep_group_1
             ↑ dp_group_0  ↑ dp_group_1  ↑ ...
```

- 每 GPU 持有 `n_experts_per_gpu = n_routing_experts / ep_size` 个 routing expert
- GPU `r`（ep_rank=r）持有 global expert id `[r·n_per_gpu, (r+1)·n_per_gpu)`
- Shared experts：每 GPU 都有**完整副本**，不走 A2A

### 4.2 每个 block 的 MoE forward 数据流

> 关键设计：SO2Convolution 内部从 Step 4 之后到 Step 5.5 之前的部分被替换为 MoE 路径。**每个 block 一对 A2A**。

```
[源 GPU] x_local: (E, F=topk+S, D_m, Cf)  # 已完成 rotate-to-local + radial mod + reshape
   │
   ▼
① 准备 routing key:
     if routing_input == "dst":
        key = type_embedding[dst]            # (E, C_type)
     elif routing_input == "src":
        key = type_embedding[src]
     elif routing_input == "src+dst":
        key = concat([type_emb[src], type_emb[dst]], dim=-1)  # (E, 2*C_type)

② Router (每 block 独立):
     topk_weights, topk_indices = router_block_i(key)
     # topk_weights: (E, topk), softmax normalized
     # topk_indices: (E, topk), values in [0, n_routing_experts)

③ Shared experts (本地, 不走 A2A):
     # x_local 的最后 S 个槽位是"占位输入"——
     # 但 shared expert 用什么作为输入？  见 §4.4 设计选择
     for s in range(S):
         sh_out_s = SharedExpert_s.run_4_layers(x_local_for_shared[:, s, :, :])
                                                       # (E, D_m, Cf)
     sh_cat = stack([sh_out_0..sh_out_{S-1}], dim=1)   # (E, S, D_m, Cf)

④ Routing expert 输入准备:
     # x_local 的前 topk 个槽位是 routing 输入
     # 注意: 每条边的 topk 个槽对应"该边 router 选中的 topk 个 expert"
     # 槽 t (t < topk) 的输入 = x_local[:, t, :, :]
     # 槽 t 的目标 expert = topk_indices[:, t]
     # 槽 t 的目标 GPU = topk_indices[:, t] // n_experts_per_gpu

     # 把 (E, topk, D_m, Cf) 展平成 (E*topk, D_m, Cf)
     routing_tokens = x_local[:, :topk, :, :].reshape(E*topk, D_m, Cf)
     expert_ids = topk_indices.reshape(E*topk)

     # 按 target_gpu 排序，记录 unsort_idx
     target_gpu = expert_ids // n_experts_per_gpu      # (E*topk,)
     sort_order = torch.argsort(target_gpu, stable=True)
     unsort_idx = torch.argsort(sort_order, stable=True)
     sorted_tokens = routing_tokens[sort_order]
     sorted_eids   = expert_ids[sort_order]
     send_counts   = torch.bincount(target_gpu, minlength=ep_size)

     # 如果 mlp_bias=True，还要打包 radial_factor 用于远端 bias_correction:
     #   原值: rad_feat_l0_focus[:, :topk, :].reshape(E*topk, Cf)  # (E*topk, Cf)
     #   打包成 (sorted_tokens, sorted_radial_factor, sorted_edge_env)
     #   一起 A2A 过去（用同一个 send_counts）

⑤ Exchange metadata: A2A 交换 send_counts ↔ recv_counts (int64)

⑥ Dispatch A2A:
     recv_tokens = _AllToAllDouble.apply(
         sorted_tokens,
         send_splits=send_counts.tolist(),
         recv_splits=recv_counts.tolist(),
         group=ep_group,
     )
     # recv_tokens: (N_recv, D_m, Cf)
     # 类似地 A2A expert_ids (不可微 int) 和 radial_factor / edge_env (如启用)

⑦ Expert compute (远端):
     # 把 recv_tokens 按 local expert id 分桶后跑 4 层 SO2Linear stack
     local_eids = recv_expert_ids % n_experts_per_gpu       # (N_recv,)
     # 用 3D 共享 tensor 模式 (n_per_gpu, in, out) 存权重
     # 对每个 local expert e:
     #   chunk_e = tokens where local_eids == e
     #   y = chunk_e
     #   for layer in [layer0..layer3]:
     #       y = SO2Linear_layer_e(y)       # (chunk, D_m, Cf) → (chunk, D_m, Cf)
     #       if layer == 0 and mlp_bias:
     #           y[:, 0, :] += bias0_e * (recv_radial_factor_chunk * recv_edge_env_chunk - 1.0)
     #       y = GatedActivation_layer(y)   # except last layer: Identity
     # cat 回 (N_recv, D_m, Cf)

⑧ Combine A2A: 用对调的 splits 把结果送回源 GPU
     returned = _AllToAllDouble.apply(
         expert_out,
         send_splits=recv_counts.tolist(),
         recv_splits=send_counts.tolist(),
         group=ep_group,
     )
     # returned: (E*topk, D_m, Cf)

⑨ Unsort + reshape + concat with shared:
     # unsort 还原原始顺序
     routing_unsorted = returned[unsort_idx]               # (E*topk, D_m, Cf)
     routing_out = routing_unsorted.reshape(E, topk, D_m, Cf)
     full = torch.cat([routing_out, sh_cat], dim=1)        # (E, F=topk+S, D_m, Cf)

⑩ Alpha 加权 (替换 cross-focus competition):
     alpha_routing = topk_weights                          # (E, topk) softmax
     alpha_shared = ones(E, S)                             # alpha=1 for shared
     alpha = torch.cat([alpha_routing, alpha_shared], dim=-1)   # (E, F)
     x_local = full * alpha[:, :, None, None]              # (E, F, D_m, Cf)

   返回 x_local 给 SO2Convolution 的 Step 6 (rotate back)
```

### 4.3 单 GPU 路径（`ep_group=None`）

- 没有 A2A，直接在本地完成所有计算
- routing experts 都在本地，按 expert id 分桶 for 循环
- shared experts 同样本地
- 计算结果 concat + alpha 加权
- **必须保证 overhead < 15%**（vs 同配置 no-MoE）

### 4.4 关键设计选择：shared expert 的输入

**问题**：x_local 有 F=topk+S 个槽，前 topk 个是 routing 输入，后 S 个是什么？

**选择 A**：x_local 在 Step 4 reshape 时就生成 F=topk+S 个槽，全部承载相同的"广播信息"——本质上是同一个上游 backbone 经过 SO2Conv pre-mix 后的 reshape。每个槽看到的输入张量内容**完全相同**（因为 backbone 节点特征只有一份）。所以 routing slot t 的输入 = x_local[:, t, :, :]，shared slot s 的输入 = x_local[:, topk+s, :, :]——它们数值上相等。

**选择 B**：x_local 只有 1 个槽，所有 expert（routing + shared）都接收这同一个输入张量。这相当于 F 维度在 MoE 入口处是 "broadcast"。

**推荐选择 A**：保持 reshape 结构对称，channels 分配自然走 radial_hidden_proj。如果 `hidden_channels = F × Cf` 是用户配置的，那么 `radial_hidden_proj(channels=C → hidden_channels=F*Cf)` 自然产出 F 个 Cf 维"分块"，每个分块就是一个槽的输入。**routing slot 和 shared slot 看到不同的 Cf 维"分块"**（虽然它们在数学上是同一上游线性投影的不同列）。

→ **v1 采用选择 A**。文档化这一点：`x_local[:, t, :, :]` (t < topk) 是 routing slot t 的输入；`x_local[:, topk+s, :, :]` (s < S) 是 shared slot s 的输入。

### v1 决议（补充）：shared expert 与 routing expert 的内部结构

shared expert 与 routing expert **使用完全相同的内部结构**：

- 同样 4 层 SO2Linear stack（layer 数由 `so2_layers` 配置决定，v1 默认 4）
- 同样的中间激活序列（`GatedActivation` 在 layer 0..n-2 后；layer n-1 后是 Identity）
- 同样的 `bias_correction` 逻辑（如 `mlp_bias=True`，layer 0 上做修正）
- 同样的 `layer_scale`（v1 共享参数，由 `layer_scale` 配置决定是否启用）
- 同样的内部 `n_focus=1` 约束

理由：shared 与 routing 的输出最终在 F 维度 concat 后做 alpha 加权，结构一致才能保证两组输出的表达能力相当；任何不一致都会让 alpha 加权的语义不明确。

实现建议（Step 3）：把 4 层 SO2Linear stack 的构建与 forward 抽成一个 helper（比如 `_ExpertSO2Stack`），shared 和 routing 各自实例化，只在外层调度上区分（是否走 A2A）。

### 4.5 bias_correction 的 A2A 打包

当 `mlp_bias=True` 时，SO2Linear 的 layer 0 在 l=0 输出上做：

```python
bias_correction = bias0 * (radial_factor * edge_env - 1.0)
x_local[:, :, 0, :] += bias_correction
```

其中：

- `bias0`：expert-local 参数，每个 expert 独立的 `(Cf,)` 向量（v1 expert 内部 n_focus=1）
- `radial_factor = rad_feat_l0_focus[:, t, :]`：源 GPU 上每条边、每个 routing slot t 的 `(Cf,)` 值
- `edge_env`：每条边的 `(1,)` 标量

实现方式：源 GPU 把 `(radial_factor * edge_env - 1.0)` 计算好（shape `(E*topk, Cf)`），用同一个 `send_counts` 通过单独的 A2A 送到远端。远端的 layer 0 直接做：

```python
y_after_layer0 = SO2Linear_layer0_e(chunk)
y_after_layer0[:, 0, :] += bias0_e * recv_radial_factor_chunk
y_after_layer0 = GatedActivation(y_after_layer0)
```

**实际上**：可以把 radial_factor 与 token concat 后一起 A2A，但因为 radial_factor 只在 layer 0 用且形状不同（`Cf` vs `D_m × Cf`），分开 A2A 更清晰。

**`mlp_bias=False` 时**：完全跳过这条路径，源 GPU 也不计算 radial_factor。SeZM 默认 `mlp_bias=False`，所以 v1 实现先把 `mlp_bias=False` 走通，再加 `mlp_bias=True` 支持。

______________________________________________________________________

## 5. 梯度同步推导

### 5.1 关键问题

为什么 routing expert 梯度的分母是 `world_size` 而不是 `dp_size`？

### 5.2 推导

考虑一个 routing expert 参数 `θ_e`（属于 expert e）。它只在 dp_size 个 rank 上有副本（同一 dp_group 内不同 ep_rank 持有不同 expert，相同 ep_rank 持有同一 expert）。

**Forward 阶段**：

- 每条边的 forward 路径是：源 GPU → A2A_dispatch → expert e（其所在 GPU）→ A2A_combine → 回源 GPU
- expert 接收的 token 来自 EP 组内所有 ep_size 个 rank（A2A 把它们路由过来）

**Backward 阶段**：

- 1st backward 沿反向走：源 GPU 的 grad → A2A_combine.backward → expert.backward → A2A_dispatch.backward → 源 GPU
- expert e 的 `θ_e.grad` 由经过它的所有 token 贡献：来自 ep_size 个 rank 的 token 都对 `θ_e` 产生梯度
- 在 expert 计算时直接积分了这些贡献——所以 `θ_e.grad` 已经包含了 `ep_size` 个 rank 的输入数据
- A2A.backward 把"输入梯度"散回去给源 GPU，但 **expert 参数梯度本身就在 expert 所在 GPU 上累积完毕**

**问题**：现在我们想要 `θ_e` 在 dp_group 内同步。dp_group 内的 dp_size 个 rank 各自跑了不同 batch 的数据，所以它们的 `θ_e.grad` 不同。

- All-reduce dp_group → sum，得到 dp_size 个 rank 的 `θ_e.grad` 之和
- 此和已经表示了 `dp_size × ep_size = world_size` 个 rank 数据的累计贡献
- 要得到 "mean over all world_size rank's data" → 除以 `world_size`

**结论**：

```python
dist.all_reduce(θ_e.grad, op=SUM, group=dp_group)
θ_e.grad.div_(world_size)  # ← 除以 world_size, 不是 dp_size!
```

### 5.3 其他参数的同步

| 参数类型                                      | 检测                                        | 同步组       | 分母           |
| --------------------------------------------- | ------------------------------------------- | ------------ | -------------- |
| Routing expert（含 SO2Linear weights + bias） | name 含 `.routing_matrix` / `.routing_bias` | **dp_group** | **world_size** |
| Router（每 block 的 router gate）             | name 含 `.moe_router.` 或 `.gate.`          | world        | world_size     |
| Shared expert                                 | name 含 `.shared_experts.`                  | world        | world_size     |
| 其他 SeZM 参数（norm, embed, fitting, etc.）  | 不含 MoE 标记                               | world        | world_size     |

DDP 的自动同步默认会把**所有**参数走 world all-reduce，这对 routing expert 是错的（routing expert 只在 dp_size 个 rank 上有副本，DDP 不应同步它们）。因此训练循环必须：

```python
with self.wrapper.no_sync():
    loss.backward()
sync_moe_gradients(model, dp_group, world_group=None, dp_size, world_size)
```

详细推导见 skill `gradient-sync-arith`。

### 5.4 参数 name 命名约定

为了让 `sync_moe_gradients` 通过名字判别，命名必须严格：

```
descriptor.blocks.0.so2_conv.moe.experts.routing_matrix
                                          ^ shape (n_per_gpu, layer, num_in, num_out)
descriptor.blocks.0.so2_conv.moe.experts.routing_bias
                                          ^ shape (n_per_gpu, num_out) 仅 layer 0 用
descriptor.blocks.0.so2_conv.moe.experts.shared_experts.0.weight_*
                                          ^ shared expert 0 的 SO2Linear 权重
descriptor.blocks.0.so2_conv.moe.router.gate.matrix
```

**`_is_routing_expert_param(name)`**：返回 `(".routing_matrix" in name) or (".routing_bias" in name)`。

______________________________________________________________________

## 6. Step-by-Step 实现计划

每个 Step 的格式：**输入文件**、**实现要点**、**UT 列表**。Phase A 全部完成才进 Phase B；Phase B 完成才进 Phase C。

### Phase A: 基础模块

#### Step 1: `_AllToAllDouble` 通信原语

**输入文件**：新建 `deepmd/pt/model/descriptor/sezm_nn/moe/a2a_ops.py`

**实现要点**：

- 从 DPA3 `deepmd/pt/model/network/moe_ep_ops.py` 整文件 copy 过来
- 改命名/import 路径
- 保留 `_a2a_raw` / `_AllToAllDouble` / `all_to_all_differentiable`
- API: `group=None` 时直通；否则 `_AllToAllDouble.apply`
- **关键不变量**：`backward` 中必须 `return _AllToAllDouble.apply(...)` 才能让二阶导工作

**UT**：

- 单卡：`source/tests/pt/test_sezm_moe_a2a_ops.py`
  - `group=None` 直通；梯度直通
- 多卡 2/4 GPU：`source/tests/pt/test_sezm_moe_a2a_ops_multigpu.py`
  - forward 数据到达对端
  - 1st backward 梯度反向 A2A 正确传播
  - **2nd backward**: `torch.autograd.grad(loss, x, create_graph=True)` 再 `.backward()` → 参数梯度完整
  - 多层链式 A2A 不死锁（模拟 3 层）

#### Step 2: `MoESO2Router`

**输入文件**：新建 `deepmd/pt/model/descriptor/sezm_nn/moe/router.py`

**实现要点**：

```python
class MoESO2Router(nn.Module):
    def __init__(self, input_dim, n_routing_experts, topk, routing_input,
                 precision, seed):
        # routing_input ∈ {"dst", "src", "src+dst"}
        # input_dim = C_type if dst/src else 2*C_type
        self.gate = MLPLayer(input_dim, n_routing_experts,
                             activation_function=None, bias=False, ...)

    def forward(self, type_emb_per_edge):
        # type_emb_per_edge: (E, input_dim)
        logits = self.gate(type_emb_per_edge)             # (E, n_routing_experts)
        topk_logits, topk_indices = torch.topk(logits, k=self.topk, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)
        return topk_weights, topk_indices
```

**调用方负责**：根据 `routing_input` 用 `src` / `dst` 索引去 type_embedding 表 gather 得到 `type_emb_per_edge`。

**UT**：

- shape: `(E, topk)`
- softmax 行和 ≈ 1
- indices ∈ `[0, n_routing_experts)`
- router gate 参数有梯度
- `create_graph=True` 二阶导通过

#### Step 3: `MoESO2ExpertCollection`

**输入文件**：新建 `deepmd/pt/model/descriptor/sezm_nn/moe/experts.py`

**实现要点**：

- 一个 expert 内部 `n_focus=1`，所以 SO2Linear 的 weight 形状简化为 `(num_l*Cin, num_l*Cout)`（m=0 块）和 `((lmax-m+1)*Cin, 2*(lmax-m+1)*Cout)`（每个 |m|>0 块）
- 用 DPA3 风格的 3D 共享 tensor：
  ```python
  self.routing_weight_m0 = nn.Parameter(
      empty(num_l * Cin, num_l * Cout, n_per_gpu)
  )  # 3D: (in, out, expert_idx)
  self.routing_weight_m = nn.ParameterList(
      [
          nn.Parameter(empty((lmax - m + 1) * Cin, 2 * (lmax - m + 1) * Cout, n_per_gpu))
          for m in range(1, mmax + 1)
      ]
  )
  self.routing_bias0 = nn.Parameter(empty(Cout, n_per_gpu))  # 仅 layer 0 用
  ```
- Layer 数 = 4（默认 `so2_layers=4`）。每层独立的 3D tensor。所以总共 `4 × (m0 + m1 + ...)` 个 routing_weight 张量。
- 命名上：用 `.routing_matrix_layer{i}_m{m}` 或类似，**name 中必须含 `.routing_matrix` 或 `.routing_bias`** 让 `_is_routing_expert_param` 能识别。
- Shared experts：`nn.ModuleList([ExpertSO2Stack(...) for _ in range(S)])` 每个是独立完整的 4 层 SO2Linear stack（n_focus=1）。
- forward 接口：
  ```python
  def forward_routing(sorted_tokens, sorted_local_eids, split_sizes,
                      sorted_radial_factor=None, sorted_edge_env=None):
      # sorted_tokens: (N_recv, D_m, Cf)
      # 对每个 local expert 跑完整 4 层 stack
      # 返回 (N_recv, D_m, Cf)

  def forward_shared(x_input_shared):
      # x_input_shared: (E, S, D_m, Cf)
      # 对每个 shared expert 在本地跑完整 4 层 stack
      # 返回 (E, S, D_m, Cf)
  ```

**UT**：

- 单 expert forward 数值正确（对比 baseline SO2Linear 串接）
- shared expert：每个独立计算结果正确
- backward 全部参数有梯度
- `create_graph=True` 二阶导通过（fp64 + gradgradcheck）

### Phase B: 单层 MoE 集成

#### Step 4: `MoESO2Convolution`

**输入文件**：新建 `deepmd/pt/model/descriptor/sezm_nn/moe/conv.py`

**实现要点**：

- 这个 class 是 SO2Convolution Step 4-Step 5.5 的 MoE 替代
- 接收 `x_local: (E, F=topk+S, D_m, Cf)` + `type_emb_per_edge` + `rad_feat_l0_focus: (E, F, Cf)` + `edge_env: (E, 1)` + `ep_group`
- 返回 `x_local: (E, F, D_m, Cf)` 给 Step 6 (rotate-back)
- 内部:
  - 一个 `MoESO2Router`
  - 一个 `MoESO2ExpertCollection`
  - 两个分支：`ep_group=None` 走单卡 for 循环；`ep_group != None` 走 A2A
- **单卡分支**伪代码：
  ```python
  def _forward_single_gpu(self, x_local, routing_key, rad_factor, edge_env):
      topk_w, topk_idx = self.router(routing_key)  # (E, topk) each

      # shared expert (本地)
      sh_input = x_local[:, self.topk :, :, :]  # (E, S, D_m, Cf)
      sh_out = self.experts.forward_shared(sh_input)  # (E, S, D_m, Cf)

      # routing expert (按 expert 分桶 for 循环)
      r_input = x_local[:, : self.topk, :, :]  # (E, topk, D_m, Cf)
      r_flat = r_input.reshape(E * topk, D_m, Cf)
      r_eids = topk_idx.reshape(E * topk)
      # 按 expert id 排序后分桶
      sort_order = torch.argsort(r_eids, stable=True)
      r_sorted = r_flat[sort_order]
      eids_sorted = r_eids[sort_order]
      counts = torch.bincount(eids_sorted, minlength=self.n_routing_experts)
      offsets = ...
      r_out_sorted = self.experts.forward_routing(
          r_sorted,
          eids_sorted,
          counts.tolist(),
          rad_factor[sort_order] if mlp_bias else None,
          edge_env[sort_order] if mlp_bias else None,
      )
      r_out_flat = r_out_sorted[torch.argsort(sort_order)]
      r_out = r_out_flat.reshape(E, topk, D_m, Cf)

      # concat + alpha
      full = torch.cat([r_out, sh_out], dim=1)
      alpha_r = topk_w
      alpha_s = torch.ones(E, S, device=...)
      alpha = torch.cat([alpha_r, alpha_s], dim=-1)
      return full * alpha[:, :, None, None]
  ```
- **多卡分支**：A2A dispatch / expert / A2A combine / unsort，按 §4.2 流程。

**UT**：

- 单卡 forward shape 正确
- 单卡 backward 所有参数有梯度
- 单卡 `create_graph=True` 二阶导通过
- 单卡 `n_routing_experts=1, topk=1, S=0` 与 baseline SO2Linear stack（n_focus=1）数值接近
- 多卡 2/4 GPU forward 不报错，shape 正确
- 多卡二阶导不死锁
- **1 GPU save ckpt → 2 GPU load → 相同输入 → 输出完全一致**

#### Step 5: 改造 `SO2Convolution`

**输入文件**：修改 `deepmd/pt/model/descriptor/sezm_nn/so2.py`（**唯一一个 v1 必须改的现有 SeZM 文件**）

**实现要点**：

- `__init__` 增加 `use_moe / moe_config / ep_group` 参数
- 所有 v1 简化的约束在此校验（raise 如不满足）：`use_moe=True` ⇒ `so2_norm=False, use_so2_attn_res="none", use_compile=False`
- 校验 `n_focus == topk + n_shared_experts`
- 校验 `n_routing_experts % ep_size == 0`
- forward 增加 `use_moe` 分支：
  ```python
  if not self.use_moe:
      # 原代码完全不动
      ...
  else:
      # Step 1-4 不变
      # Step 5 替换为 self.moe_conv(...)
      x_local = self.moe_conv(
          x_local, type_emb_per_edge, rad_feat_l0_focus, edge_env, edge_index
      )
      # Step 6-9 不变
  ```
- `routing_input` 决定 `type_emb_per_edge` 怎么构造：
  ```python
  if self.routing_input == "dst":
      type_emb_per_edge = type_embedding.index_select(0, dst)
  elif self.routing_input == "src":
      type_emb_per_edge = type_embedding.index_select(0, src)
  elif self.routing_input == "src+dst":
      type_emb_per_edge = torch.cat(
          [
              type_embedding.index_select(0, src),
              type_embedding.index_select(0, dst),
          ],
          dim=-1,
      )
  ```
- `type_embedding` 需要从外部传进 SO2Convolution（之前没传，要从 SeZMInteractionBlock 一路传下来）

**UT**：

- 单卡 MoE on/off 切换都能 forward
- `use_moe=False` 回归：与 master 行为完全一致（数值 bit-exact）
- 单卡 MoE create_graph=True 二阶导通过

### Phase C: 系统集成

#### Step 6: 进程组与梯度同步

**输入文件**：新建 `deepmd/pt/utils/sezm_moe_ep_dp.py`

**实现要点**：

- 从 DPA3 `deepmd/pt/utils/moe_ep_dp.py` 整文件 copy 过来
- 改命名：`init_ep_dp_groups`、`sync_moe_gradients`、`_is_routing_expert_param`
- `_is_routing_expert_param`：检测 `".routing_matrix"` / `".routing_bias"` in name
- `sync_moe_gradients` 关键：routing expert grad → dp_group all-reduce → `.div_(world_size)`；其他 → world all-reduce → `.div_(world_size)`

**UT** 多卡 4 GPU (EP=2, DP=2)：

- routing expert 梯度：dp_group 内一致，不同 dp_group 之间不同
- shared / router / non-MoE 参数梯度全局一致
- div\_ 后数值正确

#### Step 7: 传递 `ep_group` 到 SO2Convolution

**输入文件**：修改 `deepmd/pt/model/descriptor/sezm_nn/block.py`

**实现要点**：

- `SeZMInteractionBlock.__init__` 接收 `ep_group` 和 `moe_config`，传给 `SO2Convolution`
- `forward` 接收 `type_embedding` 参数（之前没有），传给 `SO2Convolution`

**UT**：

- block 在 MoE 配置下 forward 不报错

#### Step 8: `DescrptSeZM` 顶层配置

**输入文件**：修改 `deepmd/pt/model/descriptor/sezm.py`

**实现要点**：

- 增加配置参数：`use_moe / n_routing_experts / topk / n_shared_experts / ep_size / routing_input`
- `__init__` 中调用 `init_ep_dp_groups(ep_size)` 拿到 `ep_group / dp_group / dp_size`
- 把这些传到每个 `SeZMInteractionBlock`
- forward 时把 `extended_atype` 的 `type_embedding` 传到 block

**UT**：

- 单卡完整模型构建 + forward 成功

#### Step 9: 训练循环梯度同步

**输入文件**：修改 `deepmd/pt/train/training.py`

**实现要点**：

- 训练器接收 `use_moe_ep / ep_size` 等
- 训练循环：
  ```python
  if self.use_moe_ep:
      with self.wrapper.no_sync():
          loss.backward()
      from deepmd.pt.utils.sezm_moe_ep_dp import sync_moe_gradients

      sync_moe_gradients(model, dp_group, None, dp_size, world_size)
  else:
      loss.backward()
  ```

**UT** 多卡 4 GPU：

- 训练 1 step 不报错
- 梯度同步后数值正确

#### Step 10: Checkpoint resharding

**输入文件**：新建 `deepmd/pt/utils/sezm_moe_checkpoint.py`

**实现要点**：

- 保存：每 GPU 把自己的 local expert 命名为 global expert id `{0..n_routing_experts-1}` 中的对应区间；shared / router / non-MoE 由 rank 0 保存
- 加载：根据当前 `ep_size` 切分全局 expert tensor 给各个 rank

**UT**：

- 1 GPU save → 4 GPU load → 数值一致
- 4 GPU save → 1 GPU load → 数值一致
- 同配置 save/load → 数值完全一致

### Phase D: 端到端验证

见 §7 测试矩阵。

______________________________________________________________________

## 7. 测试矩阵

| ID  | 配置                                           | 命令                                                       | 验证目标                                                    |
| --- | ---------------------------------------------- | ---------------------------------------------------------- | ----------------------------------------------------------- |
| T1  | 1 GPU, n=4, topk=2, S=1, ep_size=1             | `pytest test_sezm_moe_e2e.py::test_single_gpu`             | forward shape、backward 所有参数有梯度、create_graph 二阶导 |
| T2  | 1 GPU save → 4 GPU (EP=4, DP=1) load           | `python save.py` then `torchrun --nproc=4 load_compare.py` | E/F/V 数值完全一致（fp64）                                  |
| T3  | 1 GPU (no MoE) vs 1 GPU MoE (n=1, topk=1, S=0) | `pytest test_sezm_moe_equivalence.py`                      | E/F 数值接近（n_per_gpu=1 时 weight 初始化对齐）            |
| T4  | 8 GPU (EP=4, DP=2)                             | `torchrun --nproc=8 test_full.py`                          | T2 + T3 在多卡下都成立                                      |
| T5  | 1 GPU MoE (ep_group=None) vs 1 GPU no-MoE      | benchmark                                                  | overhead < 15%                                              |
| T6  | 4 GPU (EP=2, DP=2)，跑力 + virial 训练 1 step  | `torchrun --nproc=4 test_2nd_deriv.py`                     | 不死锁；数值与 1 GPU 一致                                   |

每个测试都对应一个 pytest 文件。命名：`test_sezm_moe_<topic>.py` 和 `test_sezm_moe_<topic>_multigpu.py`。

### 多卡 UT 的 GPU 数量规则

- 多卡 UT 必须与单卡 UT 分开写成独立文件：`test_sezm_moe_<topic>.py` 与 `test_sezm_moe_<topic>_multigpu.py`。
- 除非测试目标明确只是 2 GPU smoke test，否则多卡 UT 至少覆盖 4 GPU。
- 当前开发环境有 8 张 GPU 时，Step 验收必须优先跑 8 GPU，并在报告中写明 `torchrun --nproc_per_node`、backend（NCCL/Gloo）和通过的 rank 数。
- 对 A2A、梯度同步、checkpoint resharding、二阶导不死锁等跨 rank 行为，2 GPU 结果只能作为 smoke test，不能替代 4/8 GPU 验收。

______________________________________________________________________

## 8. 配置 schema

`input.json` 的 `descriptor` 部分增加：

```json
{
  "descriptor": {
    "type": "SeZM",
    "channels": 64,
    "n_focus": 3,                      // 必须 == topk + n_shared_experts
    "focus_dim": 0,                    // 0 表示用 channels
    "lmax": 3, "mmax": 1, "n_blocks": 3,

    "use_moe": true,
    "n_routing_experts": 32,           // 必须能被 ep_size 整除
    "topk": 2,                          // 路由 expert 数 = F 的前 topk 个 slot
    "n_shared_experts": 1,             // F 的后 S 个 slot
    "ep_size": 4,                       // 与 world_size 须满足 world_size % ep_size == 0
    "routing_input": "dst",            // "dst" / "src" / "src+dst"

    "use_compile": false,              // MoE 时必须 false
    "so2_norm": false,                 // MoE 时必须 false
    "use_so2_attn_res": "none",         // MoE 时必须 "none"

    ...
  }
}
```

______________________________________________________________________

## 9. 关键不变量与守护

每个不变量必须在代码中以 `assert` 或 `raise ValueError` 守护：

1. `n_focus == topk + n_shared_experts`（SO2Convolution init）
1. `n_routing_experts % ep_size == 0`（MoESO2Convolution init）
1. `world_size % ep_size == 0`（init_ep_dp_groups）
1. `routing_input ∈ {"dst", "src", "src+dst"}`（MoESO2Router init）
1. `topk >= 1 and topk <= n_routing_experts`（MoESO2Router init）
1. `n_shared_experts >= 0`（MoESO2Convolution init）
1. `use_moe=True` ⇒ `use_compile=False`（DescrptSeZM init）
1. `use_moe=True` ⇒ `so2_norm=False`（SO2Convolution init）
1. `use_moe=True` ⇒ `use_so2_attn_res="none"`（SO2Convolution init）
1. A2A 调用必须用 `_AllToAllDouble.apply`（实现 review 时检查）
1. `sync_moe_gradients` 分母是 `world_size`，**不是 `dp_size`**（代码 review 时检查）
1. `_is_routing_expert_param` 命名匹配 `.routing_matrix` 或 `.routing_bias`（命名约定检查）

______________________________________________________________________

## 10. DPA3 参考代码索引

参考代码全部在 `/mnt/data_nas/zhangd/claude_space/deepmd-kit-moe/`。**禁止 import**；用 `dpa3-ref-searcher` agent 去拉关键片段。

| 用途                                              | 路径                                    | 行数 |
| ------------------------------------------------- | --------------------------------------- | ---- |
| `_AllToAllDouble` 通信原语                        | `deepmd/pt/model/network/moe_ep_ops.py` | 134  |
| `MoERouter`                                       | `deepmd/pt/model/network/moe_router.py` | 81   |
| `MoEExpertCollection` 3D 共享 tensor 模式         | `deepmd/pt/model/network/moe_expert.py` | 389  |
| `_forward_single_gpu` / `_forward_multi_gpu` 流程 | `deepmd/pt/model/network/moe_layer.py`  | 1151 |
| `init_ep_dp_groups` + `sync_moe_gradients`        | `deepmd/pt/utils/moe_ep_dp.py`          | 178  |
| `loss.backward + sync_moe_gradients` 调用         | `deepmd/pt/train/training.py` line 1147 | —    |

直接 copy 时需要做的改动：

1. 改 import 路径
1. 改 class / file 命名（前缀加 `SeZM`）
1. Expert 内部结构改为 4 层 SO2Linear stack（不是单层 MLP）
1. 输入张量形状改为 `(N, D_m, Cf)` 而不是 `(N, dim)`
1. routing key 来源改为"按边的 src/dst type embedding"

______________________________________________________________________

## 附：与 DPA3 实现的关键差异

| 维度               | DPA3 MoE                                 | SeZM MoE (本项目)                                          |
| ------------------ | ---------------------------------------- | ---------------------------------------------------------- |
| feature groups     | node / edge / angle 三路                 | 单路（x_local）                                            |
| MLP 合并打包       | M3+M4 / M5+M7 合并 + Packer 复杂 packing | 不需要                                                     |
| router 数          | 3 个（每层）                             | 1 个（每 block）                                           |
| token shape        | 2D `(N, dim)`                            | 3D `(N, D_m, Cf)`                                          |
| 每 token 输出      | 单输出后加权求和                         | 输出**保留 F=topk+S 维度**（concat shared + 加权但不求和） |
| shared expert 集成 | 加在最终求和后                           | 占用 F 维度的 S 个 slot，alpha=1                           |
| expert 内部        | 单层 MLP                                 | 4 层 SO2Linear stack + activations                         |
| 二阶导             | 同方案                                   | 同方案（`_AllToAllDouble`）                                |
| 梯度同步           | 同公式                                   | 同公式                                                     |

**核心简化**：SeZM 没有 DPA3 那种多 MLP 合并/Packer 的复杂打包；MoE 入口/出口 shape 都是 `(E, F, D_m, Cf)`，路径笔直。
