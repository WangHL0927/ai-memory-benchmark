#!/usr/bin/env python3
"""提取 Claude Code / Codex 会话；真实输入只由操作者显式指定。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re

from tools.paths import PRIVATE_DATA, ROOT, private_path


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: JSON 无效") from exc


def plain(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(plain(x.get("text", x.get("content", "[非文本内容]")))
                         if isinstance(x, dict) else plain(x) for x in value)
    return json.dumps(value, ensure_ascii=False)


def clean(text):
    # 只按已知包装识别注入，不以“看起来像指令”删除真实对话。
    # Claude 的 system-reminder、hook 包装和 Codex 的环境/权限/AGENTS 包装
    # 不属于用户发言；未带包装的普通文字保留，避免误删讨论。
    for tag in ("system-reminder", "local-command-caveat", "local-command-stdout",
                "local-command-stderr", "environment_context", "permissions",
                "user_instructions", "hook-output", "hook_output"):
        text = re.sub(rf"<{tag}(?:\s[^>]*)?>.*?</{tag}>", "", text, flags=re.S)
    text = re.sub(r"^# AGENTS\.md instructions\b.*?</INSTRUCTIONS>", "", text, flags=re.S)
    return text.strip()


def tool_output(value):
    text = plain(value)
    return text if len(text) <= 2000 else text[:2000] + f"…[截断，原长 {len(text)} 字符]"


def read_session(path, source=None):
    events = iter(jsonl(path))
    try:
        first = next(events)
    except StopIteration:
        raise ValueError("会话文件为空")
    if source is None:
        source = "codex" if first.get("type") == "session_meta" else "claude"
    if source not in ("claude", "codex"):
        raise ValueError("未知来源")
    meta = {"session_id": Path(path).stem, "source": source, "date": "", "cwd": ""}
    turns, seen = [], set()
    # 工具调用附在助手消息，工具输出附在最后一条助手消息；均不额外增加轮数。
    # Codex 的 event_msg 是 response_item 的镜像，整类忽略以避免重复。
    # Claude 仅接纳顶层 user/assistant，忽略 attachment、system、progress、
    # API 请求快照、sidechain、isMeta 及思考块；保留实际工具结果。
    def append(role, text, timestamp, message_id=None):
        if not text:
            return
        if message_id and turns and turns[-1].get("message_id") == message_id:
            turns[-1]["text"] += "\n\n" + text
        else:
            turns.append(dict(role=role, text=text, timestamp=timestamp or "", message_id=message_id))

    def attach(text):
        if turns and turns[-1]["role"] == "assistant":
            turns[-1]["text"] += "\n\n" + text
        else:
            append("assistant", text, "")

    import itertools
    for event in itertools.chain([first], events):
        typ, ts = event.get("type"), event.get("timestamp", "")
        if source == "claude":
            if event.get("cwd") and not meta["cwd"]:
                meta["cwd"] = event["cwd"]
            if event.get("sessionId"):
                meta["session_id"] = event["sessionId"]
            if typ not in ("user", "assistant") or event.get("isMeta") or event.get("isSidechain"):
                continue
            uid = event.get("uuid")
            if uid and uid in seen:
                continue
            if uid:
                seen.add(uid)
            message = event.get("message", {})
            content = message.get("content", [])
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
            texts, outputs = [], []
            for block in blocks:
                kind = block.get("type")
                if kind == "text":
                    texts.append(clean(block.get("text", "")))
                elif kind == "tool_use":
                    texts.append("工具调用：" + block.get("name", "未知") + "\n参数：" + plain(block.get("input", {})))
                elif kind == "tool_result":
                    outputs.append("工具输出：" + block.get("tool_use_id", "") + "\n" + tool_output(block.get("content", "")))
                elif kind in ("image", "document"):
                    texts.append("[非文本附件]")
            for output in outputs:
                attach(output)
            append(typ, "\n\n".join(x for x in texts if x), ts, message.get("id") if typ == "assistant" else None)
        else:
            payload = event.get("payload", {})
            if typ == "session_meta":
                meta.update(session_id=payload.get("session_id", payload.get("id", meta["session_id"])),
                            cwd=payload.get("cwd", ""), date=str(payload.get("timestamp", ts))[:10])
            if typ != "response_item":
                continue
            kind, role = payload.get("type"), payload.get("role")
            uid = payload.get("id")
            if uid and uid in seen:
                continue
            if uid:
                seen.add(uid)
            if kind == "message" and role in ("user", "assistant"):
                text = "\n\n".join(clean(x.get("text", "")) for x in payload.get("content", [])
                                     if x.get("type") in ("input_text", "output_text", "text"))
                append(role, text.strip(), ts)
            elif kind in ("function_call", "custom_tool_call"):
                text = "工具调用：" + payload.get("name", "未知") + "\n参数：" + plain(payload.get("arguments", payload.get("input", "")))
                attach(text)
            elif kind in ("function_call_output", "custom_tool_call_output"):
                attach("工具输出：" + payload.get("call_id", "") + "\n" + tool_output(payload.get("output", "")))
        if not meta["date"] and ts:
            meta["date"] = ts[:10]
    meta["project"] = Path(meta["cwd"]).name if meta["cwd"] else ""
    meta["turns"] = len(turns)
    meta["user_messages"] = sum(t["role"] == "user" for t in turns)
    return meta, turns


def length_layer(turns):
    return "≤50" if turns <= 50 else "51~150" if turns <= 150 else ">150"


def render(meta, turns):
    lines = ["# 会话记录", "", *[f"- {label}：{meta[key]}" for label, key in (
        ("会话 id", "session_id"), ("来源", "source"), ("日期", "date"),
        ("cwd", "cwd"), ("cwd 推断项目", "project"), ("总轮数", "turns"))], "- project（标注者填写）：", ""]
    previous = None
    for index, turn in enumerate(turns, 1):
        try:
            stamp = datetime.fromisoformat(turn["timestamp"].replace("Z", "+00:00"))
        except (ValueError, TypeError):
            stamp = None
        gap = False
        if previous and stamp:
            try:
                gap = (stamp - previous).total_seconds() > 7200
            except TypeError:
                pass
        if len(turns) > 300 and index > 1 and (gap or (index - 1) % 300 == 0):
            lines += [f"> 建议切点：T{index} 前（{'时间间隔超过 2 小时' if gap else '每 300 轮'}）；未自动拆分。", ""]
        # 每行正文加一个空格，避免原文中的 [Tn] 被界面误识别为轮次标题。
        lines += [f"## [T{index}] {'用户' if turn['role'] == 'user' else '助手'} {turn['timestamp']}", "",
                  "\n".join(" " + line for line in turn["text"].splitlines()), ""]
        if stamp:
            previous = stamp
    return "\n".join(lines)


def extract(path, sample_id, source=None, data_dir=None):
    if not re.fullmatch(r"[\w-]+", sample_id):
        raise ValueError("sample_id 只能含字母、数字、下划线和连字符")
    target = private_path((Path(data_dir) if data_dir else PRIVATE_DATA) / sample_id / "transcript.md")
    if target.exists():
        raise ValueError(f"拒绝覆盖已有 transcript：{target}")
    meta, turns = read_session(path, source)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(meta, turns), encoding="utf-8")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("sample_id")
    parser.add_argument("--source", choices=["claude", "codex"])
    parser.add_argument("--data-dir", type=Path, default=PRIVATE_DATA)
    args = parser.parse_args()
    try:
        meta = extract(args.input, args.sample_id, args.source, args.data_dir)
        print(f"已提取 {meta['turns']} 轮 → {args.data_dir / args.sample_id / 'transcript.md'}")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
