"""Score frozen v0.8 answer receipts and the final authorized store."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from subprocess import TimeoutExpired
import time

from tools.model.v07.cli import tokenizer
from tools.model.v08.corpus import ordered
from tools.model.v08.judge import diagnose, grade
from tools.model.v08.products import PRODUCTS, text_for
from tools.model.v08.reader import answer, compact
from tools.paths import private_path
from tools.run.proxy import settings
from tools.trial import read_json,write_json


def grade_file(card,path):
    if (path/'judgment.json').is_file():
        return
    receipt=read_json(path/'answer.json')
    if receipt['status']=='answered':
        grade(card,receipt,path/'judgment.json')


def wants_judgment(path):
    answer_path=path/'answer.json'
    return (not (path/'judgment.json').is_file()
            and (not answer_path.is_file() or read_json(answer_path)['status']=='answered'))


def attempt(label, job):
    try:
        job()
        print(f'{label} complete',flush=True)
        return None
    except (RuntimeError,ValueError,TimeoutError,TimeoutExpired,OSError,KeyError) as exc:
        print(f'{label} failed {type(exc).__name__}',flush=True)
        return dict(label=label,kind=type(exc).__name__,detail=str(exc)[:300])


def evaluate(root,pilot=False,skip=()):
    _,groups=ordered(root)
    stage=root/('pilot' if pilot else 'formal')
    config=settings()
    count,_=tokenizer()
    sets={g:read_json(root/'cards'/f'{g}.json') for g in groups}
    cards={q['id']:q for g,data in sets.items() for q in (data['cards'][:1] if pilot and g in ('tc11','daily-1')
                 else [] if pilot else data['cards'])}
    by_id={i['id']:i for items in groups.values() for i in items}
    jobs=[]
    for card in cards.values():
        for mode in ('no-memory','source-session'):
            dirname=stage/'baselines'/card['id']/mode
            if not wants_judgment(dirname):
                continue
            def baseline(card=card,mode=mode,dirname=dirname):
                if not (dirname/'answer.json').is_file():
                    transcripts=[text_for(root,by_id[s]) for s in sorted(card['sources'])]
                    memory='\n\n'.join(transcripts) if mode=='source-session' else ''
                    answer(config,card,dirname,count,exported=memory)
                grade_file(card,dirname)
            jobs.append((card['id']+'/'+mode,baseline))
    for path in sorted(stage.glob('*/*/tasks/*/*/answer.json')):
        qid=path.parents[1].name
        if qid in cards and path.parent.name in ('N','E') and wants_judgment(path.parent):
            jobs.append((str(path.relative_to(stage)),
                         lambda path=path,qid=qid:grade_file(cards[qid],path.parent)))
    for file in sorted(stage.glob('*/*/*-export.json')):
        group=file.name.removesuffix('-export.json')
        if group not in groups:
            continue
        selected=[q for q in cards.values() if q['group']==group]
        if not selected:
            continue
        if pilot:
            for card in selected:
                target=file.parent/'tasks'/card['id']/'E'
                if (target/'answer.json').is_file():
                    continue
                def scoped_export(card=card,target=target,file=file):
                    memory=read_json(file)
                    answer(config,card,target,count,exported=compact(memory) if memory else '')
                    grade_file(card,target)
                jobs.append((str(target.relative_to(stage)),scoped_export))
        units=sets[group]['units']
        local_items=groups[group][:1] if pilot else groups[group]
        if pilot:
            units=[u for u in units if all(e['sample']==local_items[0]['id'] for e in u['evidence'])]
        elif group in ('tc11','tc12') and file.parents[1].name in ('X2','ai-memory'):
            # The coding account can surface legitimate shared entries derived from either project.
            local_items=groups['tc11']+groups['tc12']
        source=[dict(id=item['id'],transcript=text_for(root,item)) for item in local_items]
        path=file.with_name(f'{group}-diagnosis.json')
        if not path.is_file():
            jobs.append((str(path.relative_to(stage)),
                         lambda selected=selected,units=units,file=file,source=source,path=path:
                           diagnose(selected,units,read_json(file),source,path)))
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(attempt,label,job) for label,job in jobs if label not in skip]
        failures=[v for f in futures if (v:=f.result())]
    write_json(stage/'judge-failures.json',failures)
    return failures


def follow(root):
    def finished():
        markers=[root/f'formal-{product}.exit' for product in PRODUCTS]
        markers.extend(p.with_suffix('.exit') for p in root.glob('formal-recovery-*.pending'))
        return all(marker.is_file() for marker in markers)

    skipped=set()
    while not finished():
        skipped.update(row['label'] for row in evaluate(root,skip=skipped))
        time.sleep(45)
    return evaluate(root)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    phase=p.add_mutually_exclusive_group()
    phase.add_argument('--pilot',action='store_true')
    phase.add_argument('--follow',action='store_true')
    a=p.parse_args()
    root=private_path(a.root)
    raise SystemExit(bool(follow(root) if a.follow else evaluate(root,a.pilot)))


if __name__=='__main__':
    main()
