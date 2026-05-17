# SeZM MoE Harness — 部署说明

这是 SeZM MoE 并行实现工程的 harness 文件集（即"启动 Claude Code 工作所需的所有 md 和 agent 配置"）。它**不含任何实现代码**——所有代码由 Claude Code 在按照 `SPEC.md` 工作时产出。

## 文件清单

```
harness/
├── README.md                                  ← 你正在读
├── CLAUDE.md                                  ← Claude Code 顶层入口配置
├── SPEC.md                                    ← 完整工程规范（设计 + 维度 + Step 计划）
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

总共 9 个 markdown 文件，约 1900 行。

## 部署到工作目录

```bash
WORK_DIR=/mnt/data_nas/zhangd/claude_space/deepmd-kit-modern

# 复制顶层文件
cp CLAUDE.md SPEC.md "$WORK_DIR/"

# 复制 .claude 目录（如果工作目录已有 .claude/，注意别覆盖现有内容）
mkdir -p "$WORK_DIR/.claude/agents" "$WORK_DIR/.claude/skills"
cp -r .claude/agents/*.md "$WORK_DIR/.claude/agents/"
cp -r .claude/skills/* "$WORK_DIR/.claude/skills/"
```

## 启动 Claude Code

在工作目录下：

```bash
cd /mnt/data_nas/zhangd/claude_space/deepmd-kit-modern
claude   # 或你常用的启动方式
```

Claude Code 启动时会自动加载 `CLAUDE.md`。让它按 `CLAUDE.md` §9 的优先级阅读文档。

## 首次启动建议的 prompt

```
请先按 CLAUDE.md §9 列出的优先级把所有文档读完，然后告诉我：
(a) 你对 SPEC.md §6 的 10 个 Step 的实现顺序的理解
(b) 你识别出的高风险点（特别是 §9 的 12 条不变量中你认为最容易出错的 3 条）
(c) 你计划如何调用子 agent（sezm-moe-implementer / multi-gpu-tester / dpa3-ref-searcher）
读完后等我确认再开始 Phase A Step 1。
```

## 验证 harness 是否安装正确

让 Claude Code 跑：

```
请验证 harness 安装：
1. 列出 CLAUDE.md 中所有提到的子 agent 和 skill
2. 用 Read 工具打开每一个，确认能访问
3. 报告任何缺失
```

预期回复：所有 3 个 agent + 4 个 skill 都能正常读取。

## 重要提醒

- 工作目录是一份代码 copy，**禁止任何 git 命令**（CLAUDE.md §2.1）。
- 所有交互用中文，但代码 / 注释 / 屏幕输出全英文。
- 上层 `../.claude/skill/` 还有 `env-setup` 和 `python-dev` 两个通用 skill，本 harness 直接 reference 它们，不重复内容。

## 修改 harness

如果在工作中发现 SPEC 需要更新（比如某个 Step 的约束需要放宽），原则：

1. 修改 `SPEC.md` 中的对应章节
1. 更新 `CLAUDE.md` 的对应"陷阱"或"规则"条目
1. 在涉及的 skill 中加更新说明
1. **不要直接改 agent 配置**——agent 应该只是 SPEC 的"执行器"

修改后让 Claude Code 重新读一次更新的部分（不需要重启 session，用 `/clear` + 重新加载即可）。
