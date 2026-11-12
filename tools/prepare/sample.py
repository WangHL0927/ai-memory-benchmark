#!/usr/bin/env python3
"""枚举会话并分层抽样；默认执行提取，--list-only 只输出清单。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re

from tools.paths import PRIVATE_DATA, ROOT, private_path
from tools.prepare.extract import extract, length_layer, read_session


def discover(claude_root, codex_root):
    return [("claude", p) for p in sorted(Path(claude_root).glob("*/*.jsonl"))] + [
        ("codex", p) for p in sorted(Path(codex_root).glob("**/rollout-*.jsonl"))]


def inventory(paths, min_users=3, min_sentence=20):
    records, errors, seen = [], [], set()
    for source, path in paths:
        try:
            meta, turns = read_session(path, source)
            # 机械会话：真实用户消息不足阈值，或没有长度严格大于阈值的句子。
            # 按中文/英文句末标点及换行切句，忽略首尾空白，工具消息不算用户消息。
            sentences = [s.strip() for t in turns if t["role"] == "user"
                         for s in re.split(r"[。！？!?\n]", t["text"])]
            eligible = meta["user_messages"] >= min_users and any(len(s) > min_sentence for s in sentences)
            key = source, meta["session_id"]
            if key in seen:
                continue
            seen.add(key)
            records.append(dict(meta, path=str(path.resolve()), eligible=eligible,
                                length_layer=length_layer(meta["turns"])))
        except (ValueError, OSError, TypeError, KeyError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    return records, errors


def choose(records, count=25, seed=20261111):
    if count < 0:
        raise ValueError("抽样数量不能为负")
    rng = random.Random(seed)
    buckets = [[r for r in records if r["eligible"] and r["length_layer"] == layer]
               for layer in ("≤50", "51~150", ">150")]
    for bucket in buckets:
        rng.shuffle(bucket)
    # 三层轮流取一条；某层不足时将余额分给其余层，不重复抽取。
    selected = []
    while len(selected) < count and any(buckets):
        for bucket in buckets:
            if bucket and len(selected) < count:
                selected.append(bucket.pop())
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--min-users", type=int, default=3)
    parser.add_argument("--min-sentence", type=int, default=20)
    parser.add_argument("--claude-root", type=Path, default=Path.home() / ".claude/projects")
    parser.add_argument("--codex-root", type=Path, default=Path.home() / ".codex/sessions")
    parser.add_argument("--data-dir", type=Path, default=PRIVATE_DATA)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    try:
        target = private_path(args.data_dir)
        if min(args.min_users, args.min_sentence) < 0:
            raise ValueError("过滤阈值不能为负")
        records, errors = inventory(discover(args.claude_root, args.codex_root), args.min_users, args.min_sentence)
        selected = choose(records, args.n, args.seed)
        samples = [dict(r, sample_id=f"s{i:03d}", layer="random") for i, r in enumerate(selected, 1)]
        # 一次预检所有目标，避免中途覆盖先前标注依赖的 transcript。
        if (target / "samples.json").exists() or (not args.list_only and any(
                (target / r["sample_id"] / "transcript.md").exists() for r in samples)):
            raise ValueError("输出已有数据，请选择新的 --data-dir")
        target.mkdir(parents=True, exist_ok=True)
        manifest = dict(seed=args.seed, filters=dict(min_users=args.min_users, min_sentence=args.min_sentence),
                        distribution=dict(Counter(r["length_layer"] for r in records if r["eligible"])),
                        sessions=records, samples=samples, errors=errors)
        (target / "samples.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.list_only:
            for sample in samples:
                extract(sample["path"], sample["sample_id"], sample["source"], target)
        print(f"枚举 {len(records)} 段，符合过滤 {sum(r['eligible'] for r in records)} 段，抽中 {len(samples)} 段；异常 {len(errors)} 段")
        if len(samples) < args.n:
            print("提示：符合条件的会话不足，未重复抽样")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
