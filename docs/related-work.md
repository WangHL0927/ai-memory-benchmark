# 相关工作：现有评测与公开数据

本文按读取、摄取和公开会话数据三类整理相关工作。

链接检查日期：2026-11-12。规模和许可按所链接的论文、数据卡与仓库说明列出。

## 范围

1. 读取侧评测检查记忆能否支持后续问答；摄取侧评测检查会话内容是否被准确保留。
2. 下表中的摄取评测覆盖不同维度，其标签采用各自的记忆价值定义。本框架将 memory policy 与评分机制分开。
3. 本评测通过离线回放和预先标注的应记单元衡量漏记；导出记忆本身不足以确定遗漏内容。
4. 公开 coding agent 会话可作为候选输入；是否能够发布其加工产物取决于来源许可和具体内容授权。

## 读取侧

| 名称 | 测什么 | 链接 |
|---|---|---|
| LongMemEval（清洗版） | 长期多会话问答，带证据轮次标注 | [GitHub](https://github.com/xiaowu0162/LongMemEval) |
| BEAM | 大规模长期记忆问答，带评分 rubric | [GitHub](https://github.com/mohammadtavakoli78/BEAM) |
| LoCoMo | 长对话问答。论文版 50 段，公开发布的是 10 段子集，引用时要写明版本；CC-BY-NC-4.0 | [GitHub](https://github.com/snap-research/locomo) |
| 运行框架：Basic Memory benchmarks、Supermemory MemoryBench、OmniMemEval | 对多个记忆系统统一跑 benchmark | [BM](https://github.com/basicmachines-co/basic-memory/tree/main/benchmarks)、[MemoryBench](https://github.com/supermemoryai/memorybench)、[OmniMemEval](https://github.com/MemTensor/OmniMemEval) |

## 摄取侧

| 名称 | 覆盖的维度 | 与本项目的差距 |
|---|---|---|
| [HaluMem](https://arxiv.org/html/2511.03506) | 每个 session 的应记 memory points、更新、干扰项；分开评抽取和更新 | 生活类英文合成对话；它的 Target Precision 只在匹配上的子集里算；数据 CC-BY-NC-ND-4.0。本项目为它写了一份政策和转换器，见 [HaluMem 使用说明](halumem.md) |
| [PerMemBench](https://arxiv.org/html/2605.25535) | 会话级的“存还是不存”（F1、FNR、FPR） | 粒度是整个会话，不能判断其中哪一句该记 |
| [MCB](https://aclanthology.org/2026.nora-1.2/) | 复用时该持久化、只本地用、需核实还是要追问 | 评的是复用时的决定，不是首次摄取 |
| [DyadMem](https://arxiv.org/html/2610.03020v1) | 摄取与更新金标，详细的抽取规则 | 会收录顺带提到的稳定用户事实，与“只记明确意图”的政策不同 |
| [Memory Tax](https://arxiv.org/html/2610.34073v1) | 从原文出发，检查信息是否被完整保留到记忆中 | 只看保留是否完整，不管该不该记 |
| [The Memory Trap](https://arxiv.org/html/2611.07554v1) | 一次性、局部的许可被写成长期、通用许可 | 只覆盖许可范围这一个维度 |

本项目框架里借用了上面几项的具体方法，每一处都在 [框架附录 A](framework.md#附录-a外部引用) 和正文的 `〔参考：…〕` 标注里写明了出处。

## 社区评测项目

截至 2026-11-12，GitHub 上相关社区项目的公开说明：

| 项目 | 评测侧 | 方法概要 |
|---|---|---|
| [vectorize-io/agent-memory-benchmark](https://github.com/vectorize-io/agent-memory-benchmark) | 读取，附速度与 token 成本 | 摄取后检索、生成答案，LLM 对照标准答案判分；另有“必须返回 / 不得返回”的检索精度测试 |
| [techtheist/knowledgedrift](https://github.com/techtheist/knowledgedrift) | 读取与维护 | 生成式虚构知识库；检索、拒答、过时内容、矛盾、删除等场景；离线规则判分 |
| [marklubin/lens-benchmark](https://github.com/marklubin/lens-benchmark) | 读取与整合策略 | 按时间流式输入片段，在检查点提问；LLM 逐事实判分 |
| [stompy-ai/stompy-benchmark](https://github.com/stompy-ai/stompy-benchmark) | 对编码任务的影响 | 比较有无记忆时 coding agent 的成本、速度和任务得分 |
| [wontopos/glasshouse](https://github.com/wontopos/glasshouse) | 读取 | 结果发布规则与草案数据集，尚无结果 |
| [Fengzdadi/memory-bench-registry](https://github.com/Fengzdadi/memory-bench-registry) | — | 记忆基准登记表，按记忆类型和复现程度分类 |

上表列出的主要评分对象是读取、维护或任务表现，与本评测的摄取内容指标分开理解。

## 公开的 coding agent 会话

| 名称 | 规模 | 有没有真人发言 | 许可 | 用法建议 |
|---|---|---|---|---|
| [SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat) | 17,968 段会话 | 有 | ODC-BY，需要申请访问 | 不采用：需要申请访问 |
| [ACT-2.5B](https://huggingface.co/datasets/submit2331/ACT-2.5B) | 目前公开 183 段预览 | 有，带发布授权声明 | ODC-BY | 小规模筛选后标注，有中文 |
| [Trace Commons](https://huggingface.co/datasets/trace-commons/agent-traces) | 30 段左右的捐赠会话 | 有 | CC-BY-4.0 | 规模小，可以逐段检查；本项目使用 |
| [CodeChat V2.0](https://huggingface.co/datasets/Suzhen/CodeChat-V2.0) | 网页编程助手对话 | 有 | 见数据卡 | 补中文和多用户分布，但没有仓库和工具上下文 |
| SWE-agent / OpenHands / SWE-Gym 轨迹 | 数万条 | 没有，`user` 角色是环境反馈 | 各自不同 | 只能测“从工具输出里提取事实” |
| LMSYS-Chat-1M | 100 万段 | 有 | 协议禁止向第三方转移 | 不宜把原文打包进评测集 |

使用条件：

- **页面公开不等于原文可以再发布。** ODC-BY 只管数据库这一层，不管其中的内容；有的协议明确禁止转移。金标可以公开，不代表原文可以随金标一起发布。
- 一些“会话数据集”的公开版去掉了对话文本，只剩元数据（例如 TraceLab、GitHub Copilot coding agent 的公开 traces），用不了“该记什么”的标注。
