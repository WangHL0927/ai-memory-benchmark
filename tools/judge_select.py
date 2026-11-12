"""判定模型选型：同一批样本换不同模型/推理强度重判，与参照判定（如 Opus high）比一致率和指标偏差。

setup：在 tmp/judge-select/<配置>/data 下用符号链接搭样本（transcript、gold、指定系统的 run）。
compare：读各配置的 judgments/，与参照的 judgments-<ref>/ 对齐比较，输出 Markdown 表和 JSON。
"""
import argparse
from pathlib import Path

from tools.calibration import agreement
from tools.score_split import counts, metrics
from tools.trial import output_path, read_json, write_json
from tools.paths import TMP
from tools.model.judge_split import memories

SELECT_ROOT = TMP / 'judge-select'


def setup(source, system, ref, configs):
    samples = sorted(p.parent.parent for p in Path(source).glob(f'*/judgments-{ref}/{system}.json'))
    for config in configs:
        for sample in samples:
            target = SELECT_ROOT / config / 'data' / sample.name
            (target / 'runs').mkdir(parents=True, exist_ok=True)
            for name in ('transcript.md', 'gold.json', f'runs/{system}.json'):
                link = target / name
                if not link.exists():
                    link.symlink_to((sample / name).resolve())
    return len(samples)


def pair(ref, other, key, field):
    left = {v[key]: v[field] for v in ref}
    right = {v[key]: v[field] for v in other}
    ids = sorted(left.keys() & right.keys())
    return [left[i] for i in ids], [right[i] for i in ids]


def compare(source, system, ref):
    reference = {p.parent.parent.name: p for p in Path(source).glob(f'*/judgments-{ref}/{system}.json')}
    rows = []
    for config_dir in sorted(p for p in SELECT_ROOT.iterdir() if (p / 'data').is_dir()):
        labels = dict(recall=([], []), grounded=([], []), error=([], []))
        totals = dict(ref={}, cfg={})
        done = 0
        for name, ref_path in reference.items():
            path = config_dir / 'data' / name / 'judgments' / f'{system}.json'
            if not path.exists():
                continue
            done += 1
            r, c = read_json(ref_path), read_json(path)
            for kind, key, field in (('recall', 'unit_id', 'verdict'), ('grounded', 'memory_id', 'grounded')):
                a, b = pair(r['recall' if kind == 'recall' else 'precision'], c['recall' if kind == 'recall' else 'precision'], key, field)
                labels[kind][0].extend(a); labels[kind][1].extend(b)
            err = lambda v: v['grounded'] == 'no' or v['distorted']
            a = {v['memory_id']: err(v) for v in r['precision']}; b = {v['memory_id']: err(v) for v in c['precision']}
            for i in sorted(a.keys() & b.keys()):
                labels['error'][0].append(a[i]); labels['error'][1].append(b[i])
            sample = config_dir / 'data' / name
            gold, run = read_json(sample / 'gold.json'), read_json(sample / 'runs' / f'{system}.json')
            mem = memories(run)
            for side, judgment in (('ref', r), ('cfg', c)):
                for k, v in counts(dict(gold=gold, judgment=judgment, memories=mem), allow_legacy=True).items():
                    totals[side][k] = totals[side].get(k, 0) + v
        if not done:
            continue
        elapsed = config_dir / 'elapsed.txt'
        m_ref, m_cfg = metrics(totals['ref']), metrics(totals['cfg'])
        rows.append(dict(config=config_dir.name, samples=done, of=len(reference),
                         elapsed_s=int(elapsed.read_text().strip()) if elapsed.exists() else None,
                         recall=agreement(*labels['recall']), grounded=agreement(*labels['grounded']),
                         error=agreement(*labels['error']),
                         miss_rate=(m_cfg['miss_rate']['value'], m_ref['miss_rate']['value']),
                         false_memory_rate_by_memory=(m_cfg['false_memory_rate_by_memory']['value'], m_ref['false_memory_rate_by_memory']['value']),
                         false_memory_rate=(m_cfg['false_memory_rate']['value'], m_ref['false_memory_rate']['value'])))
    return rows


def markdown(rows, ref):
    f = lambda x: '—' if x is None else f'{x:.2f}'
    pct = lambda x: '—' if x is None else f'{x:.0%}'
    lines = [f'| 配置 | 样本 | 用时(秒) | 召回一致率 / κ | 依据一致率 / κ | 误记一致率 / κ | 漏记率（{ref} 为括号内） | 误记率（按条；{ref} 为括号内） | 误记率（按陈述，参考） |',
             '|' + '---|' * 9]
    for r in rows:
        lines.append(f"| {r['config']} | {r['samples']}/{r['of']} | {r['elapsed_s'] or '—'} | "
                     + ' | '.join(f"{pct(r[k]['raw_agreement'])} / {f(r[k]['kappa'])}" for k in ('recall', 'grounded', 'error'))
                     + f" | {pct(r['miss_rate'][0])}（{pct(r['miss_rate'][1])}） | {pct(r['false_memory_rate_by_memory'][0])}（{pct(r['false_memory_rate_by_memory'][1])}） | {pct(r['false_memory_rate'][0])}（{pct(r['false_memory_rate'][1])}） |")
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('setup', 'compare'))
    p.add_argument('--source', default='tmp/eval/halumem/data', help='2026-11-12 判定模型选型使用的数据目录')
    p.add_argument('--system', default='openviking-0.5.4.1-ltm', help='2026-11-12 判定模型选型使用的系统标识')
    p.add_argument('--ref', default='opus')
    p.add_argument('--configs', nargs='*', default=[])
    a = p.parse_args()
    if a.action == 'setup':
        print('样本数', setup(a.source, a.system, a.ref, a.configs))
    else:
        rows = compare(a.source, a.system, a.ref)
        write_json(output_path(SELECT_ROOT / 'compare.json'), rows)
        print(markdown(rows, a.ref))


if __name__ == '__main__':
    main()
