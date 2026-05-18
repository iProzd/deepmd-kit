# SeZM MoE 并行实现 — Claude Code 配置

> **代码、注释、屏幕输出统一英文；与用户交互用中文。禁止使用任何 git 命令。**

______________________________________________________________________

## 0. 任务一句话概述

为 SeZM (DPA-4) 描述符的 `SO2Convolution` 加入 **Expert Parallelism (EP) + Data Parallelism (DP)** 的 MoE 并行。MoE 的入口点在 `SO2Convolution.forward` 的 Step 4 reshape 之后（`x_local: (E, F, D_m, Cf)`），出口点在 Step 5.5（cross-focus competition）之前。**每 block 仅一次 A2A 对**。

完整工程规范见 `SPEC.md`。**在动手任何代码之前必须先读完 `SPEC.md` 全文**。

当前实现/测试进度见 `PROGRESS.md`。`SPEC.md` 只记录设计契约和验收标准，不记录日常进度。

______________________________________________________________________

## 0.5 Cursor 使用说明

本 harness 最初为 Claude Code 编写；在 Cursor 中工作时按以下映射理解旧说明：

| Claude Code 说法  | Cursor 中使用                                                                           |
| ----------------- | --------------------------------------------------------------------------------------- |
| `Task` / 子 agent | `Subagent` tool，对应 `sezm-moe-implementer` / `multi-gpu-tester` / `dpa3-ref-searcher` |
| `Read`            | `ReadFile`                                                                              |
| `Grep`            | `rg`                                                                                    |
| `Bash`            | `Shell`                                                                                 |
| `Write` / `Edit`  | `ApplyPatch` 或 Cursor 文件编辑工具                                                     |

Cursor 原生规则在 `.cursor/rules/sezm-moe.mdc`。新会话先读：`CLAUDE.md` → `SPEC.md` → `PROGRESS.md` → 相关 skill。

______________________________________________________________________

## 1. 环境

| 项目          | 路径/配置                                                                                                                           |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 工作目录      | `/mnt/data_nas/zhangd/claude_space/deepmd-kit-modern`                                                                               |
| Conda 环境    | `/mnt/data_nas/zhangd/conda_env/torch-modern`（用 `conda activate` 激活）                                                           |
| GPU           | 8 × NVIDIA（开发机）                                                                                                                |
| 训练输入      | `/mnt/data_nas/zhangd/workplace/dev26/0515_dev_dpa4_moe/multi/input.json`（**当前是非 MoE 配置，需要修改成 MoE 配置后才能跑训练**） |
| DPA3 参考代码 | `/mnt/data_nas/zhangd/claude_space/deepmd-kit-moe`（**只读参考**，不要修改）                                                        |

**环境配置/安装/激活的详细方法**：上层 `../.claude/skills/env-setup.md` 这个 skill 已经覆盖了，直接用。

**Python 常规开发流程**：上层 `../.claude/skills/python-dev.md` 这个 skill 已经覆盖了，直接用。

______________________________________________________________________

## 2. 顶层规则

1. **禁止 git**：工作目录是一份代码 copy，**任何 git 命令（status / diff / add / commit / log）都不允许**。需要看历史改动直接用 `ls` / `rg` / Glob / 文件内容对比。
1. **不支持的配置必须报错**：所有 MoE 配置组合的合法性都在初始化时检查，凡是不符合 `SPEC.md` §3 列出的"支持配置"全部 `raise ValueError(...)`，**禁止任何 fallback**。
1. **不修改 `use_moe=False` 路径**：所有现有 SeZM 行为在 `use_moe=False` 时必须**完全不变**。MoE 走一条独立的 `if self.use_moe: ...` 分支。
1. **二阶导不能丢**：力损失训练需要 `∂²E/∂x∂θ`。每个 A2A 调用必须用 `_AllToAllDouble.apply()`（递归可微）；**禁止**在 backward 中直接调原始 `dist.all_to_all_single`。详见 skill `a2a-double-backward`。
1. **梯度同步分母用 `world_size`，不是 `dp_size`**：routing expert 的梯度走 dp_group all-reduce，但**除以 world_size**（不是 dp_size）。原因详见 skill `gradient-sync-arith`。
1. **v1 强制简化**（写入配置校验）：
   - `use_moe=True` 时强制 `use_compile=False`（v1 不支持 torch.compile + MoE）
   - `use_moe=True` 时强制 `so2_norm=False`
   - `use_moe=True` 时强制 `use_so2_attn_res="none"`
   - `layer_scale` 在 MoE 模式下退化为单一形状（不 per-expert）
1. **代码位置**：MoE 相关代码全部放在 `deepmd/pt/model/descriptor/sezm_nn/moe/` 下，**新建子模块**。
1. **复用 DPA3 代码的方式**：**copy 一份过来改**，不要 `from deepmd-kit-moe.xxx import ...`。需要参考时去 `/mnt/data_nas/zhangd/claude_space/deepmd-kit-moe` 找。详见子 agent `dpa3-ref-searcher`。
1. **每个 Step 必须配套 UT**：不写 UT 不能进下一个 Step。UT 通过 → 才能集成。详见 `SPEC.md` §6 的测试矩阵。
1. **多卡 UT 用 torchrun 跑**：模板见 skill `multi-gpu-test-template`。
1. **代码风格检查**：每个 Step 完成后必须运行 `ruff check` 并修复所有问题，然后重新验证测试通过。Ruff 路径：`/root/miniconda3/bin/ruff`。
1. **多卡 UT 默认 4/8 GPU**：除非测试目标明确只适合 2 GPU smoke test，否则多卡 UT 必须至少覆盖 4 GPU；当前环境有 8 张 GPU 时，Step 验收必须优先跑 8 GPU。报告测试结果时写明实际 `torchrun --nproc_per_node` 数量和 backend。

______________________________________________________________________

## 3. 顶层数据流（必须背下来）

```
[源 GPU]
  edge_cache 构建（含 type_embedding lookup）
   │
   ▼
SO2Convolution.forward (per block):
  Step 1. pre_focus_mix(x)                      (N, D, C) 不变
  Step 2. rotate to local: bmm(D_to_m, x_src)   → (E, D_m, C)
  Step 3. radial modulation: x_local *= rad     → (E, D_m, C)
  Step 4. reshape to multi-focus:                → (E, F, D_m, Cf)   ★ MoE 入口
  ───────────────── MoE 边界 ─────────────────
  ① 计算 routing key (默认 dst 的 type_embedding)
  ② router_block_i(routing_key) → (topk_weights: (E, topk),
                                    topk_indices: (E, topk))
  ③ shared expert 本地计算（不走 A2A）:
       sh_out[s] = SharedExpert_s(x_local[:, slot_s, :, :])  (s ∈ [0, S))
                   每个 shared 槽产生一个 (E, D_m, Cf) 输出
                   最终 sh_cat = stack(sh_out_0..S-1) shape (E, S, D_m, Cf)
  ④ 取 x_local 中 routing 槽部分（前 topk 个 slot 或者整个 broadcast，
     具体打包方式见 SPEC.md §4.4）, 按 (expert_id, target_gpu) 排序 + 展开
  ⑤ A2A_dispatch → 远端 GPU 收到 (N_recv, D_m, Cf) tokens
  ⑥ 远端运行 4 层 SO2Linear(expert) + activations + bias_correction
       — 每个 expert 内部 n_focus=1
       — 注意 mlp_bias=True 时 bias_correction 需要把
         (radial_factor * edge_env - 1.0) 通过 A2A 一起送过去
  ⑦ A2A_combine → 回到源 GPU，按 unsort 索引还原
  ⑧ 与 shared expert 结果 concat:
       full = concat([routing_out (E, topk, D_m, Cf),
                      sh_cat (E, S, D_m, Cf)], dim=1)
            = (E, F=topk+S, D_m, Cf)
  ⑨ alpha 加权:
       alpha = concat([softmax(topk_logits), ones(S)], dim=-1)
             shape (E, F)
       x_local = full * alpha[:, :, None, None]
  ───────────────── MoE 边界 ─────────────────
  Step 5.5 跳过（被上面 ⑨ 替代）
  Step 6. reshape back:  (E, F, D_m, Cf) → (E, D_m, F*Cf)
  Step 7. rotate back: bmm(Dt_from_m, ·) → (E, D, hidden_channels)
  Step 7.5. amplitude rescale
  Step 8. attention / aggregation（仍在源 GPU 上）
  Step 9. post_focus_mix
   │
   ▼
[源 GPU] 后续 FFN / residual / 下一个 block
```

整个 forward 中**每个 block 只有 1 个 A2A 对**（dispatch + combine），共 `n_blocks` 个 A2A 对（默认 3）。

______________________________________________________________________

## 4. 工作流（必须严格按 Phase 顺序）

### Phase A: 基础模块（每个 Step 单卡 UT 通过才能进下一个）

- **Step 1** `_AllToAllDouble` 通信原语：从 DPA3 copy `moe_ep_ops.py` 到 `sezm_nn/moe/` 并改命名空间。UT：`group=None` 直通 + 2/4 GPU forward/backward/2nd-backward 不死锁。
- **Step 2** `MoESO2Router`：单一 router，输入 `dst` / `src` / `src+dst` 的 type_embedding。UT：shape、softmax 行和、indices 范围、router 参数有梯度。
- **Step 3** `MoESO2ExpertCollection`：每个 expert 含 4 层 SO2Linear stack + 中间激活 + bias_correction（如启用）。UT：单 expert forward 数值正确 / shared expert / backward / `create_graph=True` 二阶导。

### Phase B: 单层 MoE 集成

- **Step 4** `MoESO2Convolution`：`ep_group=None` 时走单卡 for 循环路径；`ep_group != None` 时走 A2A 路径。UT：单卡形状/梯度/二阶导；多卡 forward 不报错；ckpt save → 多卡 load 数值一致。
- **Step 5** 改造 `SO2Convolution`：增加 `use_moe` 分支 + 所有配置校验（v1 简化的强制约束都在此处 raise）。UT：MoE on/off 切换；`use_moe=False` 回归测试不变。

### Phase C: 系统集成

- **Step 6** `init_ep_dp_groups` + `sync_moe_gradients`：从 DPA3 copy 到 `deepmd/pt/utils/sezm_moe_ep_dp.py` 改命名空间。UT 4 GPU：routing expert 梯度只在 dp_group 一致，其他参数全局一致。
- **Step 7** 修改 `SeZMInteractionBlock`：接收 `ep_group` / `moe_config` 并向下传给 `SO2Convolution`。
- **Step 8** 修改 `DescrptSeZM` 顶层：增加 MoE 配置参数（`use_moe`, `n_experts`, `topk`, `n_shared_experts`, `ep_size`, `routing_input`）。
- **Step 9** 修改 `training.py`：`use_moe_ep=True` 时 `loss.backward()` 包在 `wrapper.no_sync()` 里然后调用 `sync_moe_gradients()`。
- **Step 10** Checkpoint resharding：保存时每 GPU 存 local expert + global expert index；加载时支持 1 GPU → N GPU 的重切分。

### Phase D: 端到端正确性验证

- **T1**：1 GPU MoE forward/backward/二阶导
- **T2**：1 GPU save → 4 GPU (EP=4, DP=1) load → 相同输入 → energy/force 完全一致
- **T3**：1 GPU (no MoE) vs 1 GPU MoE (n_experts=1, topk=1, S=0) → 数值接近
- **T4**：8 GPU (EP=4, DP=2) 与上述对比正确性
- **T5**：1 GPU MoE 单卡 overhead < 15%（vs 同配置 no-MoE）
- **T6**：多卡力→virial 二阶导**不死锁**且数值与单卡一致

每个测试都对应一个 `source/tests/pt/test_sezm_moe_*.py`，多卡版本后缀 `_multigpu.py`。

______________________________________________________________________

## 5. 提交节奏

完成一个 Step → 运行该 Step 的 UT → 全绿才进下一个。

> 如果某个 Step 的 UT 不能全绿，**禁止进入下一个 Step**。先 debug。

每个 Step 完成或验证失败后更新 `PROGRESS.md`，记录已改文件、测试命令、结果和 blocker。

在 Phase B 之前不要碰 SeZM 任何描述符代码；Phase B 开始才允许改 `so2.py` 等。

______________________________________________________________________

## 6. 子 agent 与 skill 索引

- **子 agent**（用 Task tool 调用）

  - `sezm-moe-implementer`：按 SPEC 写每个 Step 的代码 + 配套 UT
  - `multi-gpu-tester`：跑 `torchrun --nproc=N` UT 并解析结果
  - `dpa3-ref-searcher`：去 `/mnt/data_nas/zhangd/claude_space/deepmd-kit-moe` 找参考代码片段

- **skill**（按需用 Read 自己读）

  - `a2a-double-backward`：递归可微 A2A 的实现规则
  - `gradient-sync-arith`：`/world_size` vs `/dp_size` 的推导
  - `sezm-moe-design`：F=topk+S 等关键设计模式
  - `multi-gpu-test-template`：torchrun UT 脚手架模板

______________________________________________________________________

## 7. 常见陷阱（提前预警）

1. **`_AllToAllDouble.backward` 直接调原始 a2a**：必须再次 `.apply()` 自己，否则二阶导断。
1. **`make_fx` 下高级索引**：在二阶导路径上用 `index_select` 而不是 `x[idx]`。（SeZM 本身已遵守）
1. **梯度同步分母**：routing expert 用 dp_group all-reduce **但分母仍是 world_size**。
1. **`n_focus == topk + n_shared_experts` 必须在初始化时校验**：v1 不接受任何其他配置。
1. **`use_compile=True` + `use_moe=True` 必须 raise**：v1 互斥。
1. **`mlp_bias=True` 时 `bias_correction` 需要把 `radial_factor * edge_env - 1.0` 通过 A2A 送到远端**：详见 SPEC §4.5。
1. **shared expert 输入是 x_local 中的哪个 slot**：v1 设计是"shared 槽位 s 用 x_local 中 routing 槽全部信息聚合"或者"用 broadcast 的同一个输入"——具体见 SPEC §4.4，**实现前必须明确这一点**。
1. **routing key 用 `dst`、`src`、`src+dst` 的 type_embedding**：是 `index_select` 的 src/dst index 去 type_embedding 表查，每条边各一份；不是 center-atom level。
1. **每 block 独立 router 不能共享**：3 个 block → 3 个独立 `MoESO2Router` 实例。

______________________________________________________________________

## 8. 训练命令模板（参考）

需要 MoE 配置后才能跑。先去 input.json 加：

```json
"descriptor": {
  "type": "SeZM",
  ...
  "use_moe": true,
  "n_routing_experts": 32,
  "topk": 4,
  "n_shared_experts": 0,
  "ep_size": 4,
  "routing_input": "dst",
  "use_compile": false   // 必须 false
}
```

启动：

```bash
cd /mnt/data_nas/zhangd/workplace/dev26/0515_dev_dpa4_moe/multi
torchrun --nproc_per_node=8 \
    /mnt/data_nas/zhangd/conda_env/torch-modern/bin/dp \
    --pt train input.json --skip-neighbor-stat
```

______________________________________________________________________

## 9. 阅读优先级

第一次接手时按以下顺序阅读，全部读完再动手：

1. **本文件 `CLAUDE.md`**（你现在在读）
1. **`SPEC.md`**（必读，含完整设计与维度推导）
1. **`PROGRESS.md`**（当前实现/验证状态）
1. `.claude/skills/sezm-moe-design/SKILL.md`
1. `.claude/skills/a2a-double-backward/SKILL.md`
1. `.claude/skills/gradient-sync-arith/SKILL.md`
1. 通过 `dpa3-ref-searcher` 拉 DPA3 的 `moe_ep_ops.py`、`moe_router.py`、`moe_expert.py`、`moe_ep_dp.py` 阅读
1. 阅读 SeZM 现有 `deepmd/pt/model/descriptor/sezm_nn/so2.py`（重点 Step 4-Step 5.5）

之后开始 Phase A Step 1。
