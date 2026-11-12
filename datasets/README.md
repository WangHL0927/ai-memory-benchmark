# 外部数据集清单

每个外部数据集一份清单（框架第 12.1 节）：来源、**固定的 revision**、所用文件的 sha256、许可、访问条件。原始数据不随仓库分发，下载到被忽略的 `tmp/datasets/<名称>/raw/`，并按清单校验哈希。

| 清单 | 许可 | redistribution | 加工产物能否进仓库 |
|---|---|---|---|
| [trace-commons.json](trace-commons.json) | CC-BY-4.0 | `allowed` | 能，需署名 |
| [halumem.json](halumem.json) | CC-BY-NC-ND-4.0 | `scores_only` | 不能，只公布分数和哈希 |

`redistribution` 是发布工具的强制检查字段：`allowed` 允许署名后发布脱敏加工产物；`scores_only` 仅发布汇总报告及样本产物 sha256，不发布 transcript、gold、runs 和 judgments。自有或私有会话不得借用外部清单发布。
