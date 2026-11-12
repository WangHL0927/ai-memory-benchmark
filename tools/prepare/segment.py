"""按 extract 标出的建议切点（框架 4.5 节：超过 300 轮或时间间隔超过 2 小时）把长会话拆成多个样本。

每段是一个独立样本：样本 id 加 -p<序号>，会话 id 不变（评分时按同一会话聚类），轮次从 T1 重新编号，
头部写明“第 k/m 段、对应原 T a~b”。原来的整段目录在拆分成功后删除（它可以由 extract 重新生成）。
"""
import argparse
import re
import shutil
from pathlib import Path

from tools.paths import private_path

CUT = re.compile(r'^> 建议切点：T(\d+) 前')
TURN = re.compile(r'^## \[T(\d+)\](.*)$')


def split(text):
    """返回 [(原起始轮次, 原结束轮次, 正文行)]；没有切点时返回 None。"""
    lines = text.splitlines()
    first = next(i for i, line in enumerate(lines) if TURN.match(line))
    header, body = lines[:first], lines[first:]
    cuts = sorted({int(m[1]) for m in (CUT.match(l) for l in body) if m})
    if not cuts:
        return header, None
    segments, current = [], []
    pending = cuts[0]
    for line in body:
        if CUT.match(line):
            continue
        turn = TURN.match(line)
        # 切点顺延到下一个用户轮之前：每段都从用户发言开始（回放型产品要求，也让每段语义完整）
        if turn and int(turn[1]) >= pending and '用户' in turn[2] and current:
            segments.append(current)
            current = []
            pending = next((c for c in cuts if c > int(turn[1])), float('inf'))
        current.append(line)
    segments.append(current)
    result = []
    for seg in segments:
        numbers = [int(m[1]) for m in (TURN.match(l) for l in seg) if m]
        result.append((numbers[0], numbers[-1], seg))
    return header, result


def render(header, index, total, start, end, lines):
    out = []
    for line in header:
        if line.startswith('- 总轮数：'):
            out.append(f'- 总轮数：{end - start + 1}')
            out.append(f'- 分段：第 {index}/{total} 段（原 T{start}~T{end}）')
        else:
            out.append(line)
    for line in lines:
        turn = TURN.match(line)
        out.append(f'## [T{int(turn[1]) - start + 1}]{turn[2]}' if turn else line)
    return '\n'.join(out).rstrip() + '\n'


def segment_dir(sample):
    header, parts = split((sample / 'transcript.md').read_text(encoding='utf-8'))
    if not parts or len(parts) < 2:
        return []
    created = []
    for k, (start, end, lines) in enumerate(parts, 1):
        target = sample.parent / f'{sample.name}-p{k}'
        if target.exists():
            raise ValueError(f'目标已存在：{target}')
        target.mkdir()
        (target / 'transcript.md').write_text(render(header, k, len(parts), start, end, lines), encoding='utf-8')
        created.append(target.name)
    shutil.rmtree(sample)
    return created


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args()
    root = private_path(args.data_dir)
    for sample in sorted(p.parent for p in root.glob('*/transcript.md')):
        if (sample / 'gold.json').exists():
            continue  # 已经标注过的样本不再拆分
        created = segment_dir(sample)
        if created:
            print(f'{sample.name} → {", ".join(created)}')


if __name__ == '__main__':
    main()
