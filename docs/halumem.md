# HaluMem 使用说明

本文说明 HaluMem 在本评测中的用法：原始数据格式、转换方式、与官方指标的关系和许可。入库规则见 [policies/halumem-v1.md](../policies/halumem-v1.md)；数据集清单（固定 revision 与 sha256）见 [datasets/halumem.json](../datasets/halumem.json)。

## 原始格式与标签

官方发布的是 **JSONL，每行一个用户对象**，不是每行一个 memory point。根对象有 `uuid`、`persona_info`、`sessions`。每个 session 包含 `start_time`、`end_time`、`dialogue_turn_num`、`dialogue`、`memory_points`、`questions`、`dialogue_token_length`，以及官方代码会读取、但 README 和数据卡没有列出的 `is_generated_qa_session`（`eval/evaluation.py` 第 51 行）。对话消息为 `role`、`content`、`timestamp`、`dialogue_turn`；公开示例一问一答的 `dialogue_turn` 同为 0。[HF 数据卡](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem)、[官方适配器的 JSONL 读取](https://github.com/MemTensor/HaluMem/blob/main/eval/eval_memzero.py)

每个 memory point 的公开字段：

| 原字段 | 含义与本地映射 |
|---|---|
| `index` | session 内点编号；原样放在单元 `halumem` 元信息中 |
| `memory_content` | 已标注的记忆描述；原样作为 `statement` |
| `memory_type` | 三种内容类别；原样作为 `category` |
| `memory_source` | 原始来源标签：`primary`、`secondary`、`interference`、`system`；保留，不臆造其细分生成规则 |
| `is_update` | 官方样例是字符串 `"True"` / `"False"`；转成属性 `"true"` / `"false"`，也接受 JSON 布尔值 |
| `original_memories` | 更新所涉及的旧记忆文本列表，不是原始对话引文；保留更新链，不能凭它生成缺席的对话 |
| `importance` | 0～1 的重要性数值；精确数值保留于 `halumem.importance`，不作为属性评分 |
| `timestamp` | 记忆创建/更新时间；保留原文，不假定它就是首次出现的消息时间 |

`memory_source: interference` 是 distractor，转入 N，原因 `distractor`；其余来源转入 U（本 session 已明确被替换的点除外）。**干扰项不是“低 importance”，也不是 `is_update` 的另一种值。**[数据卡字段说明](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem)、[官方评分中的 interference 分支](https://github.com/MemTensor/HaluMem/blob/main/eval/evaluation.py)

公开格式**没有 memory point → 原始消息轮次的 evidence 字段**。session 归属由所在 `sessions` 元素确定。转换器按对话消息顺序编号 T1、T2……，与本框架 `extract.py` 一致；不直接使用从 0 开始且可能重复的 `dialogue_turn`。编号对应表放在 `gold.halumem.source_dialogue_turns`，原始 session 下标另存 `session_index`。同一源用户的多个 session 共用其 uuid 作为 `gold.session_id`，因此 bootstrap 按源用户聚类。转换器写入 `gold.layer=random`，干扰项负例的 layer 为 targeted；样本是否经过随机抽取应由实际取样过程说明。

## 与 HaluMem 官方指标的对应关系

本评测的分数**不能直接与 HaluMem 论文公布的百分比比较**：两者的判定方式、粒度和运行条件都不同。

| 官方指标 | 官方定义 | 与本评测的关系 |
|---|---|---|
| Memory Recall | 完整性得 2 分的记忆点 ÷ 非干扰记忆点总数 | 点集、粒度和判定一致时，等于 1 − 漏记率。本评测的 partial 不计半分，完全没覆盖与部分覆盖分开报告 |
| Weighted Recall | 按 importance 加权的完整性得分 | 本评测不计算；gold 中保留了 `halumem.importance`，可自行另算 |
| Target Memory Precision | 只在命中目标的提交记忆子集里计算准确性 | 无对应项。本评测误记率的分母是全部陈述，不只取命中子集 |
| Memory Accuracy | 全部提交记忆的准确性得分（0/1/2）均值 | 无对应项。本评测按陈述二元判定，官方按整条记忆三档打分 |
| FMR（False Memory Resistance） | 成功忽略干扰项的比例 | 干扰项在本评测中转为负例，目前不报告负例指标 |
| Extraction F1 | Recall 与 Target Precision 的调和平均 | 本评测不报告 F1 |

官方定义见论文 §5.1 和官方代码（[论文](https://arxiv.org/html/2511.03506)、[evaluation.py](https://github.com/MemTensor/HaluMem/blob/main/eval/evaluation.py)、[eval_tools.py](https://github.com/MemTensor/HaluMem/blob/main/eval/eval_tools.py)）。运行条件上还有以下差异：

1. **写入方式**：官方按用户时间线连续写入，每个 session 后立即评估；本评测每段会话从空库开始，不保留跨 session 的记忆状态，也不评更新任务和问答。
2. **样本范围**：官方会跳过 `is_generated_qa_session`，并把部分更新点按系统的检索结果移入更新评估；本评测导出全部原始 session，金标不随系统输出变化。
3. **失败处理**：官方区分 `(all)` 与 `(valid)` 两种聚合；本评测把失败样本列为缺失，不计入分母。

要建立数值对照，需要取得相同系统的逐 session 输出和官方的逐点判定结果，先核对计数与分母，再核对百分比。

## 转换

```sh
# 输入为另行下载的本地 JSONL，输出目录必须尚不存在
python3 -m tools.prepare.halumem tmp/datasets/halumem/raw/HaluMem-Medium.jsonl
# Long 使用单独的目录，不与 Medium 混合
python3 -m tools.prepare.halumem tmp/datasets/halumem/raw/HaluMem-Long.jsonl --data-dir tmp/datasets/halumem/converted-long
```

- **产物**：`tmp/datasets/halumem/converted/<文件名前缀>-<uuid 哈希>-sNNNN/` 下的 `transcript.md` 和 `gold.json`。转换器不下载数据，也不运行被测系统。
- **金标来源**：gold 的 `origin: blind` 表示标签来自系统运行前的外部金标，不表示本项目重新盲标或复核过。
- **证据粒度**：每个记忆点的证据引用所属 session 的全部消息（`evidence_alignment=session_context`），可以追溯，但不是逐点精确对齐。
- **防泄露**：`persona_info`、`questions`、记忆点和更新链都不写进 transcript，避免把答案交给被测系统。
- **拒绝条件**：字段异常、未知类别、重复的 uuid 或 index、缺少更新链、输出目录已存在，或单个 session 超过 300 条消息时，转换器直接报错，不静默跳过。

## 许可证与真实数据准备

HF 数据卡声明 **CC-BY-NC-ND-4.0**。本项目只将其用于非商业评测，对外只公布分数和产物哈希，**不发布转换后的 transcript 和 gold**。仓库中的测试材料全部为自编，不含数据集内容。[HF 许可证声明](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem)

数据集提供 `HaluMem-Medium.jsonl`（约 33.5 MB）和 `HaluMem-Long.jsonl`（约 107 MB），2026-11-12 的报告使用 Medium；Long 的上下文条件不同，使用时单独报告。下载按 [datasets/halumem.json](../datasets/halumem.json) 固定的 revision 进行，并校验 sha256。[HF 文件列表](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/tree/main)
