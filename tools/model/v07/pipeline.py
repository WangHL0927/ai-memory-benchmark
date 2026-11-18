"""Resume v0.7 annotation or evaluation from private frozen artifacts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.model.v07.evaluate import BASELINES, evaluate
from tools.model.v07.protocol import annotate
from tools.paths import private_path
from tools.trial import now, read_json, write_json


def model_work(root, stage, concurrency, only_products=None):
    root = private_path(root)
    folders = sorted(p.parent for p in root.glob('*/source.json'))
    if not folders:
        raise ValueError('No registered samples')
    if stage == 'annotate':
        jobs = [(folder, None) for folder in folders]
    else:
        jobs = []
        for folder in folders:
            gold = read_json(folder / 'gold.json')
            names = {p.stem for p in (folder / 'runs').glob('*.json') if not p.stem.endswith('-drafts')}
            if only_products:
                names &= set(only_products)
            elif gold['tasks']:
                names |= BASELINES
            jobs.extend((folder, name) for name in sorted(names))
    def worker(job):
        folder, name = job
        stamp = dict(stage=stage, sample_id=folder.name, system=name)
        print(f'{now()} 开始 {stage} {folder.name} {name or ""}', flush=True)
        try:
            if stage == 'annotate':
                annotate(folder)
            else:
                evaluate(folder, name)
        except Exception as exc:
            print(f'{now()} 失败 {stage} {folder.name} {name or ""} {type(exc).__name__}', flush=True)
            return dict(stamp, status='failed', error_type=type(exc).__name__, detail=str(exc)[:400])
        print(f'{now()} 完成 {stage} {folder.name} {name or ""}', flush=True)
        return dict(stamp, status='ok')
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        receipts = list(executor.map(worker, jobs))
    write_json(root / f'{stage}-status.json', receipts)
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('annotate', 'evaluate'))
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--systems', nargs='+')
    args = parser.parse_args()
    rows = model_work(args.data_dir, args.stage, args.concurrency, args.systems)
    return int(any(r['status'] == 'failed' for r in rows))


if __name__ == '__main__':
    raise SystemExit(main())
