# SeZM MoE Harness — Cursor / Claude Code 说明

这是 SeZM MoE 并行实现工程的 harness 文件集：规范、进度、agent/skill 配置，以及 Cursor 原生规则。最初版本面向 Claude Code，现在也适配 Cursor agent/subagent 工作流。

`SPEC.md` 是设计契约，不记录日常进度；当前实现状态记录在 `PROGRESS.md`。

## 文件清单

```
harness/
├── README.md                                  ← 你正在读
├── CLAUDE.md                                  ← 顶层工程规则（Cursor 也会读取）
├── SPEC.md                                    ← 完整工程规范（设计 + 维度 + Step 计划）
├── PROGRESS.md                                ← 当前实现/验证进度
├── .cursor/
│   └── rules/
│       └── sezm-moe.mdc                       ← Cursor 原生 always-apply 规则
└── .claude/
    ├── agents/
    │   ├── sezm-moe-implementer.md            ← 子 agent：按 SPEC 写代码 + UT
    │   ├── multi-gpu-tester.md                ← 子 agent：跑 torchrun UT
    │   └── dpa3-ref-searcher.md               ← 子 agent：去 DPA3 参考代码里找模式
    └── skills/
        ├── a2a-double-backward/SKILL.md       ← 二阶可微 A2A 的实现规则
        ├── gradient-sync-arith/SKILL.md       ← /world_size vs /dp_size 推导
        ├── sezm-moe-design/SKILL.md           ← F=topk+S 等关键设计模式
        └── multi-gpu-test-template/SKILL.md   ← torchrun UT 脚手架模板
```

核心原则：`SPEC.md` 管设计，`PROGRESS.md` 管状态，agent/skill 只做执行约束和工具说明。

## 部署到工作目录

```bash
WORK_DIR=/mnt/data_nas/zhangd/claude_space/deepmd-kit-modern

# 复制顶层文件
cp README.md CLAUDE.md SPEC.md PROGRESS.md "$WORK_DIR/"

# 复制 Cursor / Claude 配置（如果工作目录已有配置目录，注意别覆盖现有内容）
mkdir -p "$WORK_DIR/.cursor/rules"
cp .cursor/rules/*.mdc "$WORK_DIR/.cursor/rules/"
mkdir -p "$WORK_DIR/.claude/agents" "$WORK_DIR/.claude/skills"
cp -r .claude/agents/*.md "$WORK_DIR/.claude/agents/"
cp -r .claude/skills/* "$WORK_DIR/.claude/skills/"
```

## 在 Cursor 中使用

Cursor 会读取本目录中的规则和上下文。开始新任务时，先确认：

- 已读 `CLAUDE.md`、`SPEC.md`、`PROGRESS.md`
- 严格不使用任何 git 命令
- 交互用中文；代码、注释、报错和命令输出用英文
- 需要子任务时使用 Cursor Subagent：
  - `dpa3-ref-searcher`：只读搜索 DPA3 参考实现
  - `multi-gpu-tester`：运行 torchrun 多卡测试
  - `sezm-moe-implementer`：按 `SPEC.md` 单 Step 实现代码和 UT

Cursor 工具名与旧 Claude Code 文档中的工具名大致对应：

- `Read` → `ReadFile`
- `Grep` → `rg`
- `Bash` → `Shell`
- `Task` → `Subagent`
- `Write/Edit` → `ApplyPatch` 或 Cursor 文件编辑工具

## 启动 Claude Code（兼容）

在工作目录下运行 Claude Code 时，仍按原流程加载 `CLAUDE.md`：

```bash
cd /mnt/data_nas/zhangd/claude_space/deepmd-kit-modern
claude
```

## 首次启动建议的 prompt

```
请先按 CLAUDE.md §9 列出的优先级把所有文档读完，然后告诉我：
(a) 你对 SPEC.md §6 的 10 个 Step 的实现顺序的理解
(b) 你识别出的高风险点（特别是 §9 的 12 条不变量中你认为最容易出错的 3 条）
(c) 你计划如何调用子 agent（sezm-moe-implementer / multi-gpu-tester / dpa3-ref-searcher）
(d) 你从 PROGRESS.md 识别出的当前进度和下一个最小动作
读完后等我确认再开始 Phase A Step 1。
```

## 验证 harness 是否安装正确

让 agent 跑：

```
请验证 harness 安装：
1. 列出 CLAUDE.md / README.md 中所有提到的子 agent 和 skill
2. 用 Read 工具打开每一个，确认能访问
3. 检查 PROGRESS.md 是否存在
4. 报告任何缺失
```

预期回复：所有 3 个 agent、4 个项目 skill、上层 env/python skill、`PROGRESS.md` 都能正常读取。

## 重要提醒

- 工作目录是一份代码 copy，**禁止任何 git 命令**（CLAUDE.md §2.1）。
- 所有交互用中文，但代码 / 注释 / 屏幕输出全英文。
- 上层 `../.claude/skills/` 还有 `env-setup` 和 `python-dev` 两个通用 skill，本 harness 直接 reference 它们，不重复内容。
- `torch-modern` 环境当前能 import `torch` / `deepmd.pt` 并识别 8 张 CUDA GPU；详细快照见 `PROGRESS.md`。

## 修改 harness

如果在工作中发现 SPEC 需要更新（比如某个 Step 的约束需要放宽），原则：

1. 修改 `SPEC.md` 中的对应章节
1. 更新 `PROGRESS.md` 中的当前状态、验证结果或 blocker
1. 更新 `CLAUDE.md` / `.cursor/rules/sezm-moe.mdc` 的对应"陷阱"或"规则"条目
1. 在涉及的 skill 中加更新说明
1. agent 配置只放执行流程和工具约束，设计决策仍写回 `SPEC.md`

修改后让 Claude Code 重新读一次更新的部分（不需要重启 session，用 `/clear` + 重新加载即可）。
