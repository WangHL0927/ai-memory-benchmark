#!/usr/bin/env python3
"""把本地 HaluMem JSONL 转为内部评测样本；不联网、不调用模型、不生成 run。"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import tempfile

from tools import score
from tools.score import require
from tools.paths import TMP, private_path


POLICY = {"id": "halumem", "version": "v1"}
CATEGORIES = ("Persona Memory", "Event Memory", "Relationship Memory")
SOURCES = ("primary", "secondary", "interference", "system")


def boolean(value):
    """官方示例使用字符串 True/False；也接受真正的 JSON 布尔值。"""
    require(type(value) is bool or value in ("True", "False"), "is_update 必须为 True/False")
    return value is True or value == "True"


def importance(value):
    """评分器只支持字符串枚举；精确数值始终另存于 halumem.importance。

    百分位网格内使用规范十进制字符串；网格外用 other_precision，绝不四舍五入。
    该回退只影响属性质量检查的精细度，不改变原始加权召回权重。
    """
    require(type(value) in (int, float) and 0 <= value <= 1, "importance 必须为 0～1 的有限数值")
    number = Decimal(str(value))
    if number * 100 != (number * 100).to_integral_value():
        return "other_precision"
    return format(number.normalize(), "f")


def make_sample(user, session, sid, prefix, frozen_at, policy):
    require(isinstance(session, dict), "session 必须为对象")
    dialogue = session.get("dialogue")
    points = session.get("memory_points")
    require(isinstance(dialogue, list) and dialogue, "dialogue 必须为非空列表")
    require(isinstance(points, list), "memory_points 必须为列表")
    turns, source_turns = [], []
    previous = None
    for message in dialogue:
        require(isinstance(message, dict) and message.get("role") in ("user", "assistant")
                and isinstance(message.get("content"), str) and bool(message["content"].strip()),
                "dialogue 消息须含 user/assistant role 和非空 content")
        source_turn = message.get("dialogue_turn")
        require(type(source_turn) is int and source_turn >= 0, "dialogue_turn 必须为非负整数")
        require(previous is None or source_turn >= previous, "dialogue_turn 必须按原顺序递增或相等")
        source_turns.append(source_turn)
        previous = source_turn
        # 官方一问一答可共享 dialogue_turn；本框架与 extract.py 一样按消息编号。
        turns.append((len(turns) + 1, message))
    require(len(source_turns) <= 300, "单 session 超过 300 轮：请先人工按话题/时间切段并核对 memory_points")

    # JSONL 每行是一个用户；每个 session 独立成样本。uuid 哈希避免路径注入。
    user_key = hashlib.sha256(user["uuid"].encode("utf-8")).hexdigest()[:16]
    sample_id = f"{prefix}-{user_key}-s{sid + 1:04d}"
    # 同一用户的 session 共用 session_id，使框架 bootstrap 按用户聚类。
    gold = dict(sample_id=sample_id, session_id=user["uuid"], source=f"halumem:{prefix}",
                turns=len(source_turns), layer="random", project=["halumem"],  # 样本来自数据集本身；干扰项负例仍记 targeted
                standard_version="v0.4", policy=deepcopy(POLICY), frozen_at=frozen_at,
                frozen_standard_version="v0.4", frozen_policy=deepcopy(POLICY),
                frozen_project=["halumem"], gold_revision=0, U=[], N=[], reannotations=[],
                halumem=dict(uuid=user["uuid"], session_index=sid,
                             source_dialogue_turns=source_turns,
                             dialogue_turn_num=session.get("dialogue_turn_num"),
                             dialogue_token_length=session.get("dialogue_token_length"),
                             is_generated_qa_session=session.get("is_generated_qa_session", False),
                             start_time=session.get("start_time"), end_time=session.get("end_time"),
                             evidence_alignment="session_context", unit_granularity="original_memory_point"))
    seen = set()
    replaced = set()
    for point in points:
        require(isinstance(point, dict), "memory point 必须为对象")
        if boolean(point.get("is_update")) and point.get("memory_source") != "interference":
            originals = point.get("original_memories", [])
            require(isinstance(originals, list) and all(isinstance(s, str) for s in originals),
                    "original_memories 必须为字符串列表")
            # 相同文本的无内容变化不算旧状态，避免把更新点自身变成负例。
            replaced.update(old for old in originals if old != point.get("memory_content"))
    for position, point in enumerate(points, 1):
        require(isinstance(point, dict), "memory point 必须为对象")
        index = point.get("index")
        require(type(index) is int and index not in seen, "memory point index 必须为不重复整数")
        seen.add(index)
        require(point.get("memory_type") in CATEGORIES, "未知 memory_type")
        require(point.get("memory_source") in SOURCES, "未知 memory_source")
        statement = point.get("memory_content")
        require(isinstance(statement, str) and bool(statement.strip()), "memory_content 必须非空")
        is_update = boolean(point.get("is_update"))
        weight = importance(point.get("importance"))
        originals = point.get("original_memories", [])
        require(isinstance(originals, list) and all(isinstance(s, str) for s in originals),
                "original_memories 必须为字符串列表")
        require(not is_update or bool(originals), "更新点缺少 original_memories")
        # 公开格式没有逐点 evidence/轮次。引用整个所属 session 的真实消息，
        # 不把 memory_content 冒充原话，不用词面相似度猜测首次确认轮次。
        evidence = [dict(turn=t, quote=m["content"], role=m["role"]) for t, m in turns]
        reason = "distractor" if point["memory_source"] == "interference" else (
            "superseded" if statement in replaced else None)
        negative = reason is not None
        unit = dict(unit_id=f"{sample_id}-{'n' if negative else 'u'}{position:03d}",
                    statement=statement, conditions=[], scope="project:halumem", evidence=evidence,
                    uncertain=False, origin="blind",
                    note="外部金标格式导入；证据为 session 全文，未做逐点语义对齐或复合点切分。",
                    halumem=deepcopy(point))
        if negative:
            unit.update(reject_reason=reason, layer="targeted")
            gold["N"].append(unit)
        else:
            # 原字段直接映射；is_update 字符串化；旧状态仅保留在原始更新链，
            # 不从 original_memories 凭空补 U，也不把较早 session 的合法 U 删除。
            unit.update(category=point["memory_type"], attributes=dict(
                is_update=str(is_update).lower(), memory_source=point["memory_source"]))
            gold["U"].append(unit)
    gold["blind_snapshot"] = deepcopy({"U": gold["U"], "N": gold["N"]})
    score.validate_gold(gold, policy)
    return gold, render(gold, session, turns)


def render(gold, session, turns):
    """只给被测系统对话与必要元信息；persona_info、金标和 questions 不进入输入。"""
    lines = ["# 会话记录", "", f"- 会话 id：{gold['session_id']}",
             f"- 来源：{gold['source']} / session {gold['halumem']['session_index']}",
             f"- 日期：{session.get('start_time', '未提供')}", "- cwd：不适用",
             "- cwd 推断项目：不适用", f"- 总轮数：{gold['turns']}",
             "- project（标注者填写）：halumem", "",
             "> 切段点：原始 session 边界；T 从 1 编号，对应表仅存 gold 元信息。", ""]
    for turn, message in turns:
        role = "用户" if message["role"] == "user" else "助手"
        lines += [f"## [T{turn}] {role} {message.get('timestamp', '')}", ""]
        # 与 extract.py 相同，缩进原文，防止正文伪造轮次标题。
        lines += ["\n".join(" " + line for line in message["content"].split("\n")), ""]
    return "\n".join(lines)


def convert(input_path, data_dir=None, *, prefix=None, frozen_at=None):
    """读取一份官方 JSONL；拒绝覆盖目录，先完整验证再落盘。

    每行 {uuid, persona_info, sessions:[{dialogue, memory_points, ...}]}。
    不读取 extracted_memories / memories_from_system，不根据系统输出修改 U。
    """
    input_path = Path(input_path)
    data_dir = Path(data_dir) if data_dir is not None else TMP / "datasets/halumem/converted"
    require(not data_dir.exists(), f"拒绝覆盖已有目录：{data_dir}")
    prefix = prefix or input_path.stem.lower()
    require(isinstance(prefix, str) and re.fullmatch(r"[a-zA-Z0-9_-]+", prefix), "prefix 仅允许字母、数字、下划线、连字符")
    frozen_at = frozen_at or datetime.now(timezone.utc).isoformat()
    policy = score.load_policy(reference=POLICY)
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    sample_ids, users = [], set()
    # 暂存目录仅用于成功后整体移入；异常时自动清理，不留半份 gold。
    with tempfile.TemporaryDirectory(prefix=".halumem-", dir=data_dir.parent) as temporary:
        staged = Path(temporary) / "samples"
        staged.mkdir()
        with input_path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    user = json.loads(line)
                    require(isinstance(user, dict) and isinstance(user.get("uuid"), str)
                            and bool(user["uuid"].strip()), "每行须含非空 uuid")
                    require(user["uuid"] not in users, "重复 uuid")
                    users.add(user["uuid"])
                    require(isinstance(user.get("sessions"), list), "sessions 必须为列表")
                    for sid, session in enumerate(user["sessions"]):
                        gold, transcript = make_sample(user, session, sid, prefix, frozen_at, policy)
                        target = staged / gold["sample_id"]
                        target.mkdir()
                        (target / "transcript.md").write_text(transcript, encoding="utf-8")
                        (target / "gold.json").write_text(json.dumps(gold, ensure_ascii=False, indent=2,
                                                                   allow_nan=False) + "\n", encoding="utf-8")
                        sample_ids.append(gold["sample_id"])
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f"{input_path}:{line_number}: {exc}") from exc
        require(bool(sample_ids), "输入没有可转换的 session")
        require(not data_dir.exists(), f"拒绝覆盖已有目录：{data_dir}")
        staged.rename(data_dir)
    return [data_dir / sid for sid in sample_ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="操作者已取得的官方 JSONL；本程序不下载")
    parser.add_argument("--data-dir", type=Path, default=TMP / "datasets/halumem/converted")
    parser.add_argument("--prefix", help="样本 id 前缀；默认输入文件名去掉扩展名")
    args = parser.parse_args()
    try:
        output = private_path(args.data_dir)  # 许可不允许再发布，按不可公开数据处理
        samples = convert(args.input, output, prefix=args.prefix)
        print(f"已转换 {len(samples)} 个样本到 {output}（仅内部评测，禁止再发布转换数据）。")
        print("证据为 session 原文；需人工复核逐点引用、条件和复合点。官方指标需另对齐判定与更新分母。")
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.exit(2, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
