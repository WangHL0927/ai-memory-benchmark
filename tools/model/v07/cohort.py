"""Freeze the v0.7 registered source transcripts without importing old judgments or labels."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from tools.config import sha256_text
from tools.paths import private_path
from tools.prepare.segment import TURN, render
from tools.trial import transcript, write_json


@dataclass(frozen=True)
class Sample:
    dataset: str
    name: str
    length: int
    cluster: str
    prefix: int = 0
    expected_hash: str = ''


SAMPLES = (
    Sample('trace-commons', 'tc18-p2', 172238, 'c99032e9'),
    Sample('trace-commons', 'tc19', 235297, 'd4c4d47d'),
    Sample('trace-commons', 'tc20-p1', 73736, 'da5d32d6'),
    Sample('trace-commons', 'tc21', 89431, 'f93127ef'),
    Sample('trace-commons', 'tc03', 13943, '11ef2190'),
    Sample('trace-commons', 'tc04-t1-50', 111310, '17c2cf98', 50, '15d873183f2f9977'),
    Sample('trace-commons', 'tc11-t1-61', 99582, '6a7964ed', 61, '1a1680ea88a04e47'),
    Sample('trace-commons', 'tc12-t1-24', 90752, '7563bddf', 24, '1ef2b6241a29f666'),
    Sample('halumem', 'halumem-medium-3091fa7ca13db45e-s0006', 13031, '348492a7'),
    Sample('halumem', 'halumem-medium-58bee016603999d0-s0017', 18155, '1b846c59'),
    Sample('halumem', 'halumem-medium-5f754dc769afc3eb-s0004', 14411, '42f702e3'),
    Sample('halumem', 'halumem-medium-7040d88d453561ed-s0015', 16073, 'bde53a03'),
)
SMOKE = 'tc21'
REDUCED = frozenset(s.name for s in SAMPLES if s.dataset == 'halumem') | {
    'tc18-p2', 'tc19', 'tc20-p1', 'tc21',
}


def clip(text, count):
    lines = text.splitlines()
    positions = [(i, m) for i, line in enumerate(lines) if (m := TURN.fullmatch(line))]
    first = positions[0][0]
    next_index, nxt = positions[count]
    if int(nxt[1]) != count + 1 or '用户' not in nxt[2]:
        raise ValueError('注册切点必须位于下一用户轮之前')
    return render(lines[:first], 1, 2, 1, count, lines[first:next_index])


def halumem_points(path):
    selected = {s.name for s in SAMPLES if s.dataset == 'halumem'}
    result = {}
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            user = json.loads(line)
            key = hashlib.sha256(user['uuid'].encode()).hexdigest()[:16]
            for n, session in enumerate(user['sessions'], 1):
                name = f'halumem-medium-{key}-s{n:04d}'
                if name in selected:
                    result[name] = session['memory_points']
    if set(result) != selected:
        raise ValueError('原始数据缺少注册的 memory points')
    return result


def prepare(history, halumem, output, only=None, reduced=False):
    output = private_path(output)
    points = halumem_points(halumem) if not only or only != SMOKE else {}
    receipts = []
    for sample in SAMPLES:
        if (only is not None and only != sample.name) or (reduced and sample.name not in REDUCED):
            continue
        parent = sample.name.split('-t1-', 1)[0] if sample.prefix else sample.name
        source = (history / sample.dataset / 'data' / parent / 'transcript.md').read_text(encoding='utf-8')
        text = clip(source, sample.prefix) if sample.prefix else source
        header, turns = transcript(text)
        cluster = header['会话 id']
        digest = sha256_text(text)
        if len(text) != sample.length or not cluster.startswith(sample.cluster) or not digest.startswith(sample.expected_hash):
            raise ValueError(f'注册输入校验失败：{sample.name}')
        folder = output / sample.name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'source-transcript.md'
        if path.exists() and path.read_text() != text:
            raise ValueError(f'已经登记的输入发生变化：{sample.name}')
        path.write_text(text)
        details = dict(sample_id=sample.name, dataset=sample.dataset, session_id=cluster,
                       turns=len(turns), source_sha256=digest, parent=parent)
        write_json(folder / 'source.json', details)
        if sample.dataset == 'halumem':
            write_json(folder / 'source-points.json', points[sample.name])
        receipts.append(details)
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--halumem-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample', choices=[s.name for s in SAMPLES])
    parser.add_argument('--reduced', action='store_true')
    args = parser.parse_args()
    receipts = prepare(args.history, args.halumem_source, args.output, args.sample, args.reduced)
    print(f'Inputs checked: {len(receipts)}')


if __name__ == '__main__':
    main()
