"""Run isolated v0.8 domains, keeping each product's observable failures."""
import argparse
from pathlib import Path

from tools.model.v07.cli import tokenizer
from tools.model.v08.corpus import ordered
from tools.model.v08.products import AiMemory, BasicMemory, ClaudeMem, OpenViking, X2, text_for
from tools.model.v08.reader import answer, compact
from tools.paths import private_path
from tools.run.proxy import settings
from tools.trial import read_json, write_json

BACKENDS={'X2':X2,'ai-memory':AiMemory,'basic-memory':BasicMemory,
          'claude-mem':ClaudeMem,'openviking':OpenViking}


def read_scope(backend,group,cards,config,count,directory,pilot=False):
    item=next(i for i in backend.items if i['group']==group)
    try:
        memory=backend.export(group)
        write_json(directory/f'{group}-export.json',memory)
    except (RuntimeError,ValueError,TimeoutError,OSError) as exc:
        memory=None
        write_json(directory/f'{group}-export-error.json',dict(issue=type(exc).__name__))
    for card in cards:
        q=directory/'tasks'/card['id']
        q.mkdir(parents=True,exist_ok=True)
        a=q/'N'
        before=len(backend.proxy.calls)
        try:
            with backend.reader(item,a) as native:
                result=answer(config,card,a,count,native=native)
        except (RuntimeError,ValueError,TimeoutError,OSError,KeyError) as exc:
            result=dict(status='native_not_tested',issue=type(exc).__name__,detail=str(exc)[:220],
                        reads=[],calls=[],delivered=[],read_tokens=None)
            write_json(a/'answer.json',result)
        write_json(a/'read-product-usage.json',backend.proxy.calls[before:])
        if memory is not None and (pilot or card['export_subset']):
            answer(config,card,q/'E',count,exported=compact(memory) if memory else '')
        print(type(backend).__name__,card['id'],'N',result['status'],len(result['reads']),result['read_tokens'],flush=True)


def product_run(root, product, pilot, only_domain=None):
    _, groups=ordered(root)
    config=settings()
    count, _=tokenizer()
    target=root/('pilot' if pilot else 'formal')/product
    lists={name:read_json(root/'cards'/f'{name}.json') for name in groups}
    domains=([('coding',[groups['tc11'][0]]),('daily-1',[groups['daily-1'][0]])] if pilot else
             [('coding',groups['tc11']+groups['tc12']),('daily-1',groups['daily-1']),('daily-2',groups['daily-2'])])
    for domain,items in domains:
        if only_domain and only_domain!=domain:
            continue
        directory=target/domain
        if (directory/'done.json').exists():
            continue
        if pilot and any(directory.glob('tasks/*/N/answer.json')):
            # An earlier local evaluation error can leave a complete native trajectory.
            write_json(directory/'done.json',dict(product=product,domain=domain,pilot=True,resumed=True))
            continue
        backend=BACKENDS[product](items,directory,config)
        group_ids=list(dict.fromkeys(i['group'] for i in items))
        with backend.proxy, backend.open():
            for item in items:
                backend.step(item,text_for(root,item))
            for group in group_ids:
                cards=lists[group]['cards']
                read_scope(backend,group,cards[:1] if pilot else cards,config,count,directory,pilot)
        write_json(directory/'done.json',dict(product=product,domain=domain,pilot=pilot))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('product',choices=tuple(BACKENDS))
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--domain',choices=('coding','daily-1','daily-2'))
    a=p.parse_args()
    if a.pilot and a.domain=='daily-2':
        p.error('daily-2 is outside the pilot')
    product_run(private_path(a.root),a.product,a.pilot,a.domain)


if __name__=='__main__':
    main()
