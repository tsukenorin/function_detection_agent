# Function Detection Agent

基于 LLM、历史 CVE 领域知识库（RAG）与版本隔离代码仓库上下文挖掘的 **C/C++ 函数级自动化漏洞检测智能体**（高级软件工程 · 课程大作业）。

## 核心特性

- **LangGraph 自主多阶段编排**：采用代码级两阶段置信度门控（第 1 阶段物理不绑定仓库工具独立初判，仅存疑且指定外部符号时才放行第 2 阶段按需探索），避免无休止过度探索与修复版误报。
- **严格成对版本隔离（红线保障）**：
  - `SampleLoader` 消费 `target` 标签：`target=1`（漏洞版）仅解析修复前父提交 `C^`，`target=0`（修复版）仅解析修复提交 `C`；进入 Agent 的 `AgentInput` 禁止携带任何标签、`commit_message` 或 `git diff`。
  - `RepoFS` 按 `{owner}/{repo}/{allowed_sha}` 物理隔离缓存只读快照，通过 `LazyWorkspace` 惰性物化，工具层以闭包绑定单一工作区，杜绝跨版本泄露与路径越界。
- **历史 CVE 领域知识库 RAG**：仅使用课程提供数据集（`data/knowledge.jsonl`，3,789 对历史 CVE），离线提炼结构化漏洞模式卡与最小修复差异 `fix_pattern`，采用代码词面 + 安全原语双通道 RRF 混合检索、CVE/同名反劫持去重与相关性阈值截断。
- **Tree-sitter 确定性渐进式调用链**：内置 `inspect_call_chain`（1~3 跳渐进前沿展开）与 `read_symbol_body`（按 C/C++ AST 精确读取函数/结构体/宏定义），配合 `grep`、`find_symbol` 等只读沙箱工具。
- **纯客观证据摘要压缩子代理**：独立短上下文子代理 `EvidenceCompactorSubAgent` 将冗长仓库输出提纯为客观事实 + 原样关键代码行摘录，防止多跳上下文稀释。
- **四要素可追溯报告**：输出 `idx`、判定结果、判定理由及系统自动采集的知识检索/仓库读取审计记录，原生兼容官方 `dataset/evaluation.py`。

## 目录结构

```text
function_detection_agent/
├── agent/                  # 核心 Schema、Protocol 接口、LangGraph 编排层、压缩子代理、应用服务
├── tool/                   # Workspace 沙箱原语、Tree-sitter 调用链、只读工具集、预算与审计日志
├── repo/                   # SampleLoader 隔离边界、GitHub 父提交解析、RepoFS 惰性快照工作区
├── knowledge/              # 历史 CVE 成对模式卡加工与双通道混合 RAG 检索器
├── experiments/            # 对比实验运行器及 12 对 / 30 对实验归档结果
├── tests/                  # 78 项单元测试与端到端隔离集成测试
└── docs/                   # 工具层设计文档、架构图与完整对比实验报告
```

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，或直接设置环境变量（默认兼容 DeepSeek / OpenAI 协议）：

```bash
DEEPSEEK_API_KEY=your_api_key
# 或使用 OpenAI 兼容服务：
# OPENAI_API_KEY=your_api_key
# OPENAI_BASE_URL=https://your-api-host.example/v1
# OPENAI_MODEL=your_model
```

### 3. 运行单元测试

```bash
.venv/Scripts/python.exe -m pytest -q
```

### 4. 运行对比实验

将课程数据集放入 `data/dataset.zip` 与 `data/knowledge.jsonl` 后运行：

```bash
.venv/Scripts/python.exe -m experiments.run_experiment \
  --model deepseek-flash \
  --pairs 30 \
  --workers 6 \
  --runs improved_rag_check,improved_v3_two_stage \
  --output-dir experiments/results_30pairs
```

## 实验效果摘要

基于课程测试集（`deepseek-flash`，`reasoning_effort="low"`），全部结果均经官方 `dataset/evaluation.py` 核验：

### 12 对（24 样本 / 9 仓库）消融演进

| 实验组 | P-C（↑） | P-V（↓） | P-B（↓） | P-R（↓） | Accuracy | 平均工具 | 平均耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline`（孤立函数） | 1 | 10 | 1 | 0 | 54.2% | 0.00 | 23.3s |
| `improved_rag_check`（混合 RAG + 自检） | 6 | 2 | 4 | 0 | 75.0% | 0.00 | 39.8s |
| `improved`（无门控仓库，5 次预算） | 3 | 2 | 7 | 0 | 62.5% | 4.62 | 70.7s |
| `improved_gated`（Prompt 门控，2 次预算） | 5 | 3 | 3 | 1 | 66.7% | 1.29 | 35.7s |
| `improved_v3_two_stage`（代码级两阶段门控 + Tree-sitter + 客观压缩） | 4 | 4 | 4 | 0 | 66.7% | 1.04 | 50.1s |

### 30 对（60 样本 / 22 仓库）头对头验证

| 实验组 | P-C（↑） | P-V（↓） | P-B（↓） | P-R（↓） | P-C 率 | Accuracy | 平均工具 | 平均耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `improved_rag_check` | 8 | 10 | 10 | 2 | 26.7% | 60.0% | 0.00 | 48.6s |
| `improved_v3_two_stage` | **10** | **7** | 10 | 3 | **33.3%** | **61.7%** | **1.05** | **43.5s** |

## 参考文档

- [工具层 v1 设计与实现](docs/2026-10-25-工具层v1设计与实现.md)
- [编排层核心接口与对比实验报告](docs/2026-11-25-编排层核心接口与对比实验报告.md)
