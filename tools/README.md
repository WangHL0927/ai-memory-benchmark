# 工具说明

本目录是评测流程的全部实现，对应[评测框架](../docs/framework.md) v0.6 的拆分判定协议（6.4 节）。本文给出各工具的用法、批处理约定和文件格式，供运行评测或接入新系统的人参考。

Python 只用标准库；报告和抽检表是不联网的单文件 HTML。所有命令都在**仓库根目录**下以 `python3 -m` 运行，`--help` 查看完整参数。

## 目录

```
tools/
  paths.py              仓库路径与写入规则：不适合公开的内容只能写进 tmp/
  config.py             读取 .env、记录运行环境与模型身份
  trial.py              共用格式、路径检查、原子写入、批处理与失败记录
  cache.py              模型结果的内容寻址缓存（只写 tmp/）
  prepare/              数据准备
    extract.py          一段 Claude Code / Codex 会话 → transcript.md
    sample.py           枚举本机会话、分层随机抽样，并调用 extract
    segment.py          按建议切点把长会话切成多个样本
    halumem.py          HaluMem JSONL → transcript.md 与 gold.json
  model/                模型代替人工加工评测集
    annotate.py         盲标：无 gold 的样本 → 冻结的 gold（不读 runs）
    judge_split.py      拆分判定：召回判断 + 精确判断
    common.py           Codex / Claude CLI 调用协议
    prompts/            带版本号的提示词
  run/                  被测系统适配器，输出 runs/<system>.json
    proxy.py            被测系统共用的本地模型代理
    openviking.py  ai_memory.py  claude_mem.py  basic_memory.py
  score_split.py        拆分判定评分：漏记率、误记率、粒度
  report_split_html.py  拆分判定报告 → 自包含静态 HTML
  calibration.py        人工抽检：抽样出表、计算一致率
  judge_select.py       判定模型选型：多配置重判并与参照比较
  derive_summary_only.py  从 OpenViking 导出中移除附带原文，生成补充对照
  publish.py            发布前扫描与版本化导出
  score.py              金标、政策校验及共用统计函数
```

## 流程与命令

以下 `<数据目录>` 的结构是 `<数据目录>/<样本 id>/transcript.md`、`gold.json`、`runs/*.json`、`judgments/*.json`。私有数据和许可受限的数据都放在 `tmp/` 下。

### 1. 数据准备

```sh
# 转换 HaluMem（默认写到 tmp/datasets/halumem/converted）
python3 -m tools.prepare.halumem tmp/datasets/halumem/raw/HaluMem-Medium.jsonl
# 提取单段会话（不覆盖已有 transcript）
python3 -m tools.prepare.extract <会话文件.jsonl> <样本 id> --source claude --data-dir <数据目录>
# 切分长会话（处理无 gold 的样本，按超过 300 轮或间隔超过 2 小时的切点拆分）
python3 -m tools.prepare.segment --data-dir <数据目录>
# 从本机会话分层随机抽样（默认 25 段，写到 tmp/private/data；--list-only 只出清单）
python3 -m tools.prepare.sample -n 25 --seed 20261111
```

### 2. 盲标

```sh
python3 -m tools.model.annotate --data-dir <数据目录> --annotator codex --model gpt-6.1-sol --reasoning-effort high
```

- 只处理有 transcript、还没有 gold 的样本，从不覆盖已有 gold。
- 输入只有框架、政策和 transcript，不读取任何 runs。
- 未传 `--policy` 时，盲标使用 default-v1；使用其他政策时显式传入政策 JSON。HaluMem 转换器已生成 gold 的样本不再盲标。
- 证据会逐条比对原文：轮次唯一可定位时自动改正，对不上的删除；删除超过 20% 的，该样本标注判为失败。


### 3. 运行被测系统

```sh
python3 -m tools.run.openviking   --data-dir <数据目录>
python3 -m tools.run.ai_memory    --data-dir <数据目录>
python3 -m tools.run.claude_mem   --data-dir <数据目录>
python3 -m tools.run.basic_memory --data-dir <数据目录>
```

- 每段会话从空记忆库开始摄取，结束后导出该系统的全部记忆，写到 `runs/<system>.json`。
- 被测系统的模型调用都经过 `run/proxy.py`。代理只校验模型名，请求原样转发；它汇总每次调用实际发出的参数和用量，写进 run 文件，并在 `tmp/` 下留存请求和响应（key 打码）。所用模型见 [README 的模型表](../README.zh-CN.md#方法)，各产品的配置见 [docs/systems.md](../docs/systems.md)。
- Basic Memory 自身不做抽取，适配器用一个固定的 agent（提示词为 Basic Memory 官方的 `memory-capture`，运行时从已安装的包中读取）决定记什么；`--max-rounds`、`--max-total-tokens` 限制这个 agent 的调用量。
- `--concurrency` 设置同时处理的样本数；每个样本使用独立的进程、端口和存储。
- `tools.prepare.segment` 生成子样本后替换原样本目录；已有 gold 的样本跳过，不重新切分。

### 4. 拆分判定

```sh
python3 -m tools.model.judge_split --data-dir <数据目录> --annotator codex --model gpt-6.1-sol --reasoning-effort low
```

- 每个 `runs/<system>.json` 生成一份 `judgments/<system>.json`，字段见下文[判定结果](#判定结果)。
- 召回判断只看应记单元和全部记忆，不看原始会话；精确判断只看一条记忆和原始会话。
- `--systems` 限定被测系统，`--force` 覆盖已有判定。
- 判定模型的选型见 [docs/judge-selection.md](../docs/judge-selection.md)。

### 5. 评分与报告

```sh
python3 -m tools.score_split --data-dir <数据目录> --output <输出路径>
python3 -m tools.report_split_html <输出路径>.json --output <输出路径>.html
```

- 输出 `<输出路径>.md` 和 `<输出路径>.json`；指标见下文[报告指标](#报告指标)。
- 报告头部每个系统一行，列出模型、提示词来源和调用次数；逐次参数和提示词哈希只写进 JSON。
- `--agreement` 附上人工抽检的一致率；不提供时，报告标为“未经人工校准”。
- 数据不在 `data/` 里时，报告只能写进 `tmp/`，因为报告明细会带出原文。

### 6. 人工抽检

```sh
# 按固定种子分层抽样，生成抽检表
python3 -m tools.calibration sample --data-dir <数据目录> -n 50 --output <抽检表.html>
# 根据人工答案计算原始一致率和 Cohen κ
python3 -m tools.calibration score <人工答案> --sheet <抽检表.json> --output <一致率.json>
```

抽检表要求先独立作答、再展开模型结果；展开后答案锁定。精确题需要填写陈述计数和是否含原文摘录，分类合计不能超过陈述总数。一致率同时比较整条标签和陈述计数，并分别给出人工与模型的误记率。

### 7. 发布

```sh
python3 -m tools.publish scan   --data-dir <数据目录> --dataset trace-commons
python3 -m tools.publish export --data-dir <数据目录> --dataset trace-commons --eval-version <版本> --report <report.json>
```

- `--dataset` 是 `datasets/` 下的清单名。`scan` 检查个人信息等风险，结果写到 `tmp/publish/`，不修改输入。
- `export` 按数据集清单的 `redistribution` 字段决定导出范围：`allowed` 导出加工产物，`scores_only` 只导出分数和产物哈希。

## 批处理约定

适用于盲标、被测运行和判定：

- **断点续跑**：盲标已有 gold 直接跳过；判定跳过有效且输入哈希匹配的结果；被测运行已有结果默认跳过，`--force` 才覆盖；失败时保留原文件。
- **日志**：进度写在 `tmp/logs/<失败记录所在目录>-<失败记录名>-<时间>.log`，每个样本开始、结束各一行，立即落盘。长任务以日志为准。
- **熔断**：连续失败 3 次就停止派发剩余样本，避免系统性错误空耗模型额度。claude-mem 适配器使用 10 次。
- **失败清单**：`--failures` 指定位置（限 `tmp/`）。清单记录样本、错误类型、尝试次数和时间，不保存密钥和模型原文。批次有失败时，退出码为 1；参数错误时为 2。
- **缓存**：模型调用按“模型 + 推理参数 + 提示词 sha256 + 输入 sha256”寻址，命中就直接复用；`--no-cache` 跳过缓存。模型原始返回留存在 `tmp/model-raw/`。
- **模型 CLI**：必须指定 `--annotator codex|claude` 和 `--model`；`--reasoning-effort` 必须是该模型支持的取值。Codex 以 `exec -m … -c model_reasoning_effort=… -s read-only` 运行，Claude 以 `-p --model … --effort …` 运行并禁用工具、不保存会话；Codex 使用只读沙箱和临时会话。`--cli-path` 可以换成测试用的假 CLI。
- **隔离边界**：Codex 的 read-only 只限制写入，不限制读取。盲标不读取 runs 依赖 CLI 遵守提示词约定，不是操作系统层面的隔离。

## 判定结果

`judgments/<system>.json` 含两部分，另附判定者身份和 run、gold、transcript 的 sha256。

**召回判断**（`recall`，每个应记单元一条）：

| 字段 | 含义 |
|---|---|
| `unit_id` | 应记单元编号 |
| `verdict` | `full` 完整 / `partial` 缺关键条件或主体 / `none` 没有 / `distorted` 写歪 |
| `memories` | 支撑该结论的记忆编号 |
| `note` | 判定理由 |

**精确判断**（`precision`，每条记忆一条）：

| 字段 | 含义 |
|---|---|
| `claims` | 陈述总数。一个陈述只表达一个可核对的事实、偏好、决定或计划；标题和纯结构文字不计；原样附带的会话原文整体计 1 个 |
| `unsupported` | 在会话里没有依据的陈述数 |
| `partial` | 部分有依据的陈述数 |
| `distorted_claims` | 写歪原意的陈述数 |
| `verbatim` | 是否含原样附带的会话原文 |
| `grounded` / `distorted` | 整条记忆的依据（`yes` / `partial` / `no`）和是否写歪，用于与整条口径对照 |
| `worth` | 整条记忆是否值得长期记（`yes` / `no` / `unclear`），独立于对错 |
| `errors` | 至多 10 项被判为无依据或写歪的陈述，每项 `{text, type, reason}` |
| `note` | 判定理由 |

每个陈述只归入一类，优先级为写歪 > 无依据 > 部分有依据，三类之和不超过 `claims`。

## 报告指标

| 指标 | 计算 | 位置 |
|---|---|---|
| 漏记率 | 召回为 partial / none / distorted 的单元 ÷ 单元总数（不含 uncertain） | 主表 |
| 误记率 | Σ(`unsupported` + `distorted_claims`) ÷ Σ`claims`；部分有依据不计入 | 主表 |
| 每段陈述数、每段记忆条数、每条平均字符数 | 粒度信息 | 主表 |
| 按条误记率 | 整条判为无依据或写歪的记忆 ÷ 记忆总数 | JSON |
| 原文摘录占比 | 含原样会话原文的记忆 ÷ 记忆总数 | JSON |
| 完全没记住率 | 召回为 none / distorted 的应记单元 ÷ 主口径单元总数 | JSON |
| 依据不足率 | `grounded=partial` 的记忆 ÷ 记忆总数，按整条记忆计 | JSON |
| 不值得记占比 | `worth=no` 的记忆 ÷ 记忆总数 | JSON |
| 含 uncertain 的口径 | 召回统计包含 uncertain 单元 | JSON |

所有比率都附分子、分母和按 `gold.session_id` 聚类的 bootstrap 区间；分母为零时为 null。

## gold 文件约定

目录结构沿用框架第 10 节。gold 的外壳字段如下（示例为合成内容）：

```json
{
  "sample_id": "s001",
  "session_id": "synthetic-session",
  "source": "synthetic",
  "turns": 6,
  "layer": "random",
  "project": ["garden", "library"],
  "standard_version": "v0.4",
  "policy": {"id": "default", "version": "v1"},
  "frozen_at": "2026-11-11T10:00:00.000Z",
  "gold_revision": 0,
  "U": [],
  "N": [],
  "blind_snapshot": {"U": [], "N": []}
}
```

- `standard_version=v0.4` 表示现行产物格式；框架文档的版本由运行环境另行记录。
- U 的字段使用框架 4.3 节的结构；条件必须含 `key`。
- N 用 `reject_reason` 代替 `category` 和 `attributes`，另含 `layer: random | targeted`。
- `prior_context` 不进 U，也不进 N。
- 模型标注的 gold 另带 `annotator` 身份（工具、模型、推理强度、提示词版本、UTC 时间）。

run 文件名（不含 `.json`）就是系统标识，被测运行写入的格式是 `{"standard_version", "policy", "system", "D": [{"draft": {"statement": …}}]}`。

## 测试

```sh
python3 -m unittest
```

`tests/fixtures/` 全部是合成数据。测试不访问外部模型或产品；模型 CLI 使用假实现，HTTP 接口由本机合成服务提供。
