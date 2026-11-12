# 评测你自己的数据

适用场景：你有一批会话（公开数据集、团队内部记录，或者自己的 coding agent 会话），想知道某个记忆系统从中“该记的漏了多少、存下的有多少是错的”。

本文按拆分判定协议（[框架 6.4 节](framework.md#64-拆分判定另一种判定协议)）一步步说明，读完可以对自己的数据跑出一份报告。各命令的完整参数见 [tools/README.md](../tools/README.md)。

## 数据目录

每一步产出的文件都放在同一个数据目录下：

```
<数据目录>/<样本 id>/
  transcript.md          # 第 1 步
  gold.json              # 第 2 步
  runs/<系统>.json        # 第 3 步
  judgments/<系统>.json   # 第 4 步
```

**数据不能公开的，数据目录一律放在 `tmp/` 下**（例如 `tmp/private/data`），这个目录被 git 忽略。只有许可允许、并且已经脱敏的数据，才放进仓库的 `data/`。

## 第 0 步：选一份入库政策

“什么该记”由入库政策决定。不同政策下得出的分数不能相互比较。

- 评“记忆系统是否只记用户明确意图下的个人与项目知识”：用 [`policies/default-v1`](../policies/default-v1.md)。
- 你的场景有自己的标准（例如客服对话只记工单相关事实）：按[框架第 2 节](framework.md#2-入库政策必须提供什么)写一份新政策，包括 `policies/<id>-<版本>.md`（说明、判例、切分示例）和 `policies/<id>-<版本>.json`（类别、属性、原因码、条件类型、范围取值）。JSON 可以参照 [`default-v1.json`](../policies/default-v1.json)，或者测试里那份刻意写得不同的 [`alternate-v7.json`](../tests/fixtures/policies/alternate-v7.json)。
- 数据集自带金标、并且有自己的定义：为它写一份忠实于原定义的政策，示例见 [`policies/halumem-v1`](../policies/halumem-v1.md)。

## 第 1 步：把会话转成 transcript.md

格式要求见[框架 3.1 节](framework.md#31-输入)：
- 每一轮带编号，写成 `## [T1]`、`## [T2]`……
- 头部写会话 id、来源、日期、cwd 和总轮数。
- 工具输出超过 2000 字符的部分截断。

- **Claude Code / Codex 会话**：

  ```sh
  python3 -m tools.prepare.extract <会话文件.jsonl> <样本 id> --source claude --data-dir <数据目录>
  python3 -m tools.prepare.segment --data-dir <数据目录>   # 切分长会话
  ```

- **其他格式**：写一个转换脚本，输出同样格式的 transcript.md。可以参照 [`tools/prepare/halumem.py`](../tools/prepare/halumem.py)：它把 HaluMem 的 JSONL 转成 transcript.md，同时把数据集自带的金标转成 gold.json。

## 第 2 步：盲标

数据集自带金标、并且已经转换成 gold.json 的，可以跳过这一步。

用模型代替人工盲标：

```sh
python3 -m tools.model.annotate --data-dir <数据目录> --annotator codex --model <标注模型> --reasoning-effort high --policy policies/<政策>.json
```

要点（详见[框架第 4、5 节](framework.md#4-应记单元)）：

- **先标再看**：这一步不看任何被测系统的输出。
- 每个单元都要有证据（轮次 + 原文引文）。条件（范围、时效、是否仅限一次等）写进单元，不另拆单元。
- 拿不准的单元标 uncertain，单独统计。
- 用模型标注的，要从中抽一部分由人工独立复核，报告一致率（第 5 步）。

## 第 3 步：运行被测系统

被测系统**只看 transcript.md**，**从空记忆库开始**摄取，结束后导出它存下的全部记忆，写成 `runs/<系统>.json`：

```json
{
  "standard_version": "v0.4",
  "policy": {"id": "default", "version": "v1"},
  "system": {"name": "<系统>", "version": "<版本>"},
  "D": [
    {"draft": {"statement": "<一条记忆的正文>"}}
  ]
}
```

- `standard_version` 和 `policy` 照抄 gold.json 里的值。
- 拆分判定只用每条记忆的 `statement`，记忆粒度不限：可以是一句话，也可以是一整页。
- 已有适配器的模块名为 `tools.run.openviking`、`tools.run.ai_memory`、`tools.run.claude_mem`、`tools.run.basic_memory`，均接受 `--data-dir <数据目录>`。其他系统需要适配器，只转换格式，不补内容、不改措辞。

## 第 4 步：判定

```sh
python3 -m tools.model.judge_split --data-dir <数据目录> --annotator codex --model <判定模型> --reasoning-effort low
```

每个系统生成一份 `judgments/<系统>.json`，包含两部分：

- **召回判断**：每个应记单元能否从该系统的全部记忆中得出。
- **精确判断**：把每条记忆拆成陈述，逐个核对在会话里有没有依据、有没有写歪。记忆里原样附带的会话原文按 1 个陈述计。

判定模型不应与被测系统使用的 LLM 相同。选型参考见 [judge-selection.md](judge-selection.md)。

## 第 5 步：评分与人工抽检

```sh
python3 -m tools.score_split --data-dir <数据目录> --output <输出路径>
python3 -m tools.report_split_html <输出路径>.json --output <输出路径>.html
```

主表给出漏记率、误记率（无依据或写歪的陈述 ÷ 陈述总数）和粒度信息，各指标附分子、分母和按 `gold.session_id` 聚类的 bootstrap 区间。辅助指标只写进 JSON，完整列表见 [tools/README.md](../tools/README.md#报告指标)。

模型标注和判定的可信度，要靠人工抽检确认：

```sh
python3 -m tools.calibration sample --data-dir <数据目录> -n 50 --output <抽检表.html>
# 人工按抽检表独立作答后：
python3 -m tools.calibration score <人工答案> --sheet <抽检表.json> --output <一致率.json>
python3 -m tools.score_split --data-dir <数据目录> --agreement <一致率.json> --output <输出路径>
```

没有附上一致率的报告会标明“未经人工校准”。数据目录在 `tmp/` 下时，报告也只能写在 `tmp/` 下，因为报告明细会带出原文。
