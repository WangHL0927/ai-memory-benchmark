[English](README.md) | 简体中文

# AI Memory Benchmark

一套评测 AI 记忆系统的方法和工具。目前评测**摄取质量**：给记忆系统喂一段真实会话，检查它存下来的记忆里，该记的漏了多少，存下的有多少是错的。

## 目录

- [背景](#背景)
- [评测什么](#评测什么)
- [2026-11-12 数据与报告](#2026-11-12-数据与报告)
- [快速开始](#快速开始)
- [运行一次评测](#运行一次评测)
- [方法](#方法)
- [数据集与许可](#数据集与许可)
- [可复现性](#可复现性)
- [仓库结构](#仓库结构)
- [路线图](#路线图)
- [许可证](#许可证)

## 背景

带长期记忆的 AI 助手（比如 coding agent 的记忆插件），会从会话里挑出内容存进记忆库。存漏了，下次就想不起来；存错了，错误会带进以后的每一次对话。

现有的记忆评测大多测读取一侧，也就是“存进去之后能不能答对问题”，很少直接检查存进去的内容本身对不对。少数测摄取的，又各自绑定了自己对“什么该记”的定义，结果没法横向比较。详见[相关工作](docs/related-work.md)。

## 评测什么

各个记忆产品都按普通用户能直接用上的默认方式安装和配置，喂进同一批会话，比较它们的摄取效果。两项主要指标：


入库政策规定哪些信息值得保留；标注是依照政策选出信息的过程。“应记单元”是标注者按入库政策从会话里标出的、可独立使用、值得长期记住的信息；“陈述”是记忆中一个可核对的事实、偏好、决定或计划。每段会话的陈述数、记忆条数和每条平均长度也一并列出。所有指标都给出分子、分母和按 `gold.session_id` 聚类的 bootstrap 置信区间。

| 指标 | 含义 |
|---|---|
| **漏记率** | 应记单元中，未能从该系统的全部记忆里完整得出的比例 |
| **误记率** | 系统存下的陈述中，在会话里没有依据或写歪原意的比例 |



设计上的三条原则：

- **评测机制和价值取舍分开。**[评测框架](docs/framework.md)只规定怎么评；“什么该记”由可替换的[入库政策](policies/)规定，每次评测都要声明用的是哪一份。
- **与被测系统无关。**所有系统走同一套输入输出接口、同一套判定和统计。
- **先盲标，后判定。**应记单元在看到任何系统输出之前就已确定，避免被系统输出带偏。

最新结果见 [2026-11-12 评测报告](reports/2026-11-12/README.zh-CN.md)。

## 2026-11-12 数据与报告

| 部分 | 状态 |
|---|---|
| [评测框架](docs/framework.md) v0.6 | 拆分判定分别核对召回与精确，见 6.4 节 |
| 入库政策 [default-v1](policies/default-v1.md)、[halumem-v1](policies/halumem-v1.md) | 分别用于项目会话与 HaluMem 记忆点 |
| HaluMem-Medium | 报告包含 30 段会话、251 个应记单元 |
| Trace Commons | 31 段，其中 18 段所有系统都已判定 |
| 被测系统适配器：OpenViking、ai-memory、claude-mem、Basic Memory | 已接入 |
| 判定模型选型（评估导出记忆的模型） | 已完成，见 [docs/judge-selection.md](docs/judge-selection.md) |
| [2026-11-12 评测报告](reports/2026-11-12/README.zh-CN.md) | HaluMem-Medium 与 Trace Commons 结果 |
| 人工抽检（校准模型标注和判定） | 未开始 |

## 快速开始

环境要求：Python 3（只用标准库，已在 3.14 上测试）。

```sh
git clone https://github.com/WangHL0927/ai-memory-benchmark.git && cd ai-memory-benchmark

# 运行单元测试（全部使用合成数据，不访问外部模型或产品）
python3 -m unittest

# 用自带的合成样例生成一份报告
python3 -m tools.score_split --data-dir tests/fixtures/evaluation --systems alpha beta --output tmp/demo/report
python3 -m tools.report_split_html tmp/demo/report.json --output tmp/demo/report.html
# 输出 tmp/demo/report.md、report.json 和 report.html
```

## 运行一次评测

完整评测需要以下外部依赖：

- 被测系统用的模型，通过 OpenAI 兼容接口调用。
- 标注和判定用的 Codex CLI 或 Claude Code CLI。
- 各被测产品自身的运行环境。

**1. 配置**

复制 `.env.example` 为 `.env`，填入模型服务地址、key、模型名和 embedding 维度。`.env` 已被 git 忽略。

**2. 准备数据**

按 [datasets/](datasets/README.md) 里的清单下载数据集，校验固定 revision 和 sha256，再转换成带轮次编号的 `transcript.md`：

```sh
python3 -m tools.prepare.halumem tmp/datasets/halumem/raw/HaluMem-Medium.jsonl
python3 -m tools.prepare.extract <会话文件> <样本 id> --source claude
python3 -m tools.prepare.segment --data-dir <数据目录>   # 按建议切点切分长会话
```

**3. 盲标**

模型代替人工，在看到任何系统输出之前，按政策标出应记单元：

```sh
python3 -m tools.model.annotate --data-dir <数据目录> --annotator codex --model gpt-6.1-sol --reasoning-effort high
```

**4. 运行被测系统**

每个系统从空库开始摄取会话，导出全部记忆：

```sh
python3 -m tools.run.openviking --data-dir <数据目录>
python3 -m tools.run.ai_memory  --data-dir <数据目录>
python3 -m tools.run.claude_mem --data-dir <数据目录>
python3 -m tools.run.basic_memory --data-dir <数据目录>
```

**5. 判定**

```sh
python3 -m tools.model.judge_split --data-dir <数据目录> --annotator codex --model gpt-6.1-sol --reasoning-effort low
```

**6. 评分与报告**

```sh
python3 -m tools.score_split --data-dir <数据目录> --output <输出路径>
python3 -m tools.report_split_html <输出路径>.json --output <输出路径>.html
```

批处理支持断点续跑：盲标跳过已有 gold；判定复用有效且输入哈希匹配的结果；被测运行默认跳过已有结果，`--force` 才覆盖。进度写在 `tmp/logs/`；默认连续失败 3 次停止派发，claude-mem 为 10 次。各工具的参数和文件约定见 [tools/README.md](tools/README.md)。

评测自己的会话数据，见 [docs/evaluate-your-data.md](docs/evaluate-your-data.md)。

## 方法

**标注与判定。**由模型代替人工完成，属于评测集的加工，不计入测试结果：

- **标注**：在看到任何系统输出之前进行（盲标），每个应记单元都附带原文证据。证据由程序逐条比对原文校验，对不上的删除。
- **判定**：采用拆分判定，分两个方向：
  - 召回判断：每个应记单元能否从该系统的全部记忆中得出。决定漏记率。
  - 精确判断：把每条记忆拆成陈述，逐个核对在会话里有没有依据、有没有写歪。决定误记率。

  误记率按陈述计，以减小导出记忆条数不同带来的影响。陈述切分和原文摘录的计数方式仍会影响结果；记忆里原样附带的会话原文整体按 1 个陈述计。字段和计算口径见 [tools/README.md](tools/README.md#判定结果)。

| 角色 | 模型 | 推理强度 |
|---|---|---|
| 被测系统 LLM（统一） | `deepseek-v4.1-flash-1017`；其余生成参数沿用各产品默认 | — |
| 被测系统 embedding（统一） | `qwen3.7-text-embedding`；claude-mem 无法指定 embedding 模型，关闭向量 | — |
| 标注 | `gpt-6.1-sol`（Codex CLI） | high |
| 判定 | `gpt-6.1-sol`（Codex CLI） | low |

判定模型的选型见 [docs/judge-selection.md](docs/judge-selection.md)。

被测系统的模型调用都经过同一个本地代理。代理只校验模型名，请求参数原样转发；它记录每次调用实际发出的参数和用量，并留存请求与响应。API key 只保存在代理进程中，被测产品接触不到。

**被测系统。**生成参数沿用产品默认；使用外部 embedding 的系统统一为 qwen；claude-mem 关闭向量功能。各产品的默认行为和本评测的实际配置见 [docs/systems.md](docs/systems.md)。另有两点说明：

- Basic Memory 自己不做抽取，“记什么”由外部 agent 决定，所以测的是“Basic Memory + 固定 agent 和提示词”的组合，结果单列，不和其他产品并排排序。
- 产品并不知道评测用的是哪份入库政策，所以“不值得记占比”只作参考，不作为主指标。

**依据与出处。**判定的三档划分、误记率的分母、盲标流程、样本量与统计方法等，凡是借用外部方法的，都在[框架附录 A](docs/framework.md#附录-a外部引用)和正文的 `〔参考：…〕` 标注里写明了出处。

## 数据集与许可

| 数据集 | 内容 | 许可 | 本仓库发布什么 |
|---|---|---|---|
| [HaluMem-Medium](datasets/halumem.json) | 多轮日常对话，自带金标（[使用说明](docs/halumem.md)） | CC-BY-NC-ND-4.0 | 只发布分数和哈希 |
| [Trace Commons](datasets/trace-commons.json) | coding agent 会话 | CC-BY-4.0 | 署名后发布加工产物（金标、系统输出、判定；尚未发布） |

- 原始数据不随仓库分发。按清单下载到被 git 忽略的 `tmp/datasets/`，并校验哈希。
- 不适合公开的内容一律放在 `tmp/`。工具会强制检查：私有会话只能写进 `tmp/`，含原文的报告也只能写进 `tmp/`。
- 只有许可允许、并通过个人信息检查的加工产物，才会进入仓库的 `data/` 和 `reports/`。

## 可复现性

- **数据**：每个数据集都固定 revision，并记录所用文件的 sha256。
- **模型**：模型调用记录可获得的模型版本、参数、输入规模、截断信息、提示词版本和哈希；无法获取的字段标为“未记录”。提示词带版本号，见 `tools/model/prompts/`。
- **缓存**：按输入内容寻址。模型、参数、提示词和输入都相同时直接复用结果，任一项改变就自动重跑；模型的原始返回同时留存，供审计。
- **中间结果**：许可允许的标注、系统输出和判定，计划随报告一起发布，别人可以直接复核或重新评分，不用重复付费调用模型。2026-11-12 报告的中间结果尚未发布。

## 仓库结构

```
CHANGELOG.md   版本变更记录
docs/          评测框架、被测系统配置、判定模型选型、HaluMem 使用说明、相关工作、评测自有数据的步骤
policies/      入库政策：每份一个 .md 说明和一个 .json 机读版本
datasets/      外部数据集清单：来源、revision、sha256、许可
tools/         Python 标准库实现，说明见 tools/README.md
  prepare/     数据转换与切段
  model/       模型标注、判定，以及版本化提示词
  run/         各被测系统的适配器和共用模型代理
tests/         单元测试，全部使用合成数据
data/          可公开的加工产物
tmp/           被 git 忽略：私有数据、外部数据集副本、缓存、日志
```

## 路线图

**近期**

- 补齐 Trace Commons 的四个被测系统结果及配套静态 HTML 报告。
- 人工抽检：从模型的标注和判定里抽样，由人独立复核，报告一致率及采用的门槛。

**之后**

- 扩大样本量，加入更多无需申请即可使用的公开数据集。
- 接入更多记忆系统，例如 Graphiti / Zep、LangMem。
- 改进人工抽检的界面。

## 许可证

- 自有代码（`tools/`、`tests/`）：[Apache-2.0](LICENSE)，另见 [NOTICE](NOTICE)；第三方原文文件保留其上游许可。
- 文档、评测框架、入库政策和报告：[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)。
- 发布的加工数据（`data/`）：CC-BY-4.0，并保留原始数据集的署名。例如 Trace Commons 的加工产物须注明 Trace Commons contributors。HaluMem 不允许发布演绎作品，本仓库只公布其分数和哈希。
