"""测试公用：路径、合成金标与提交项的构造函数。fixture 全部为合成数据。"""
import os
from pathlib import Path
import subprocess
from tools.prepare.extract import render
from tools.trial import write_json
from tools.model.judge_split import input_hashes


ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / 'tmp'  # 测试临时文件放在被忽略的 tmp/ 里
TMP.mkdir(exist_ok=True)
FIXTURES = ROOT / 'tests' / 'fixtures'
SESSIONS = FIXTURES / 'sessions'
POLICY_FIXTURES = FIXTURES / 'policies'


def unit(uid='u1', uncertain=False, origin='blind', conditions=None):
    return dict(unit_id=uid, category='decision', attributes={'basis':'confirmed'}, statement='花园项目使用蓝色水管。',
                conditions=conditions or [], scope='project:garden', evidence=[dict(turn=1, quote='以后用蓝色水管')],
                uncertain=uncertain, origin=origin, note='合成金标：用户确认的项目选择')


def gold(units=None, negatives=None, sid='s001', session='synthetic-session', layer='random'):
    return dict(sample_id=sid, session_id=session, source='synthetic', turns=6, layer=layer,
                standard_version='v0.4', policy={'id':'default','version':'v1'}, project=['garden'], frozen_at='2026-11-11T10:00:00Z', gold_revision=0,
                U=[unit()] if units is None else units, N=negatives or [])


def run(*entries, version='v0.4'):
    return dict(standard_version=version, policy={'id':'default','version':'v1'}, D=list(entries))


def entry():
    return dict(draft=dict(statement="花园项目使用蓝色水管。", category="decision", attributes={"basis": "confirmed"},
                           scope="project:garden", conditions=[], evidence=[dict(turn=1, quote="以后用蓝色水管")]))


def sample(root, name='s001', with_gold=True):
    target = Path(root) / name
    target.mkdir(parents=True, exist_ok=True)
    meta = dict(session_id='synth-' + name, source='synthetic', date='2026-11-12', cwd='/synthetic/garden', project='garden', turns=2)
    turns = [dict(role='user', text='以后用蓝色水管', timestamp=''),
             dict(role='assistant', text='好的。\n\n工具调用：echo\n参数：{}\n工具输出：成功', timestamp='')]
    (target / 'transcript.md').write_text(render(meta, turns), encoding='utf-8')
    if with_gold:
        write_json(target / 'gold.json', gold(sid=name, session='synth-' + name))
    return target


def cli(module, *args, env=None):
    env = dict(env or os.environ, EVAL_DOTENV=str(ROOT / 'tmp' / 'no-such-dotenv'))  # 不读本机 .env
    return subprocess.run(['python3', '-m', module, *map(str, args)], cwd=ROOT, text=True, capture_output=True, env=env)


def fake_cli(root, outputs):
    # 每次进程独立，调用计数及 stdin 放在测试目录，便于断言重试与盲标材料。
    root = Path(root)
    state = root / 'cli-state.json'
    responses = root / 'responses.json'
    write_json(responses, outputs)
    program = root / 'fake-cli'
    program.write_text('''#!/usr/bin/env python3
import json, sys
from pathlib import Path
root = Path(__file__).parent
if sys.argv[1:] == ['--version']:
    print('fake-cli 0.0'); sys.exit(0)  # 版本查询不计入调用次数
state = root / 'cli-state.json'
old = json.loads(state.read_text()) if state.exists() else []
old.append(dict(args=sys.argv[1:], prompt=sys.stdin.read(), cwd=str(Path.cwd())))
state.write_text(json.dumps(old, ensure_ascii=False))
responses = json.loads((root / 'responses.json').read_text())
value = responses[min(len(old)-1, len(responses)-1)]
print(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
''', encoding='utf-8')
    program.chmod(0o755)
    return program, state


def recall(uid='u1', verdict='full', refs=None):
    return dict(unit_id=uid, verdict=verdict, memories=([] if verdict == 'none' else ['m1']) if refs is None else refs, note='合成判断')

def precision(mid='m1', grounded='yes', distorted=False, worth='yes', *, claims=1,
              unsupported=None, partial=None, distorted_claims=None, verbatim=False, errors=None):
    return dict(memory_id=mid, grounded=grounded, distorted=distorted, worth=worth, note='合成判断',
                claims=claims, unsupported=int(grounded == 'no' and not distorted) if unsupported is None else unsupported,
                partial=int(grounded == 'partial' and not distorted) if partial is None else partial,
                distorted_claims=int(distorted) if distorted_claims is None else distorted_claims,
                verbatim=verbatim, errors=[] if errors is None else errors)


def fixture(root, name='s001', system='alpha', units=None, verdicts=None, precision_values=None, session=None, metadata=None):
    s = sample(root, name)
    g = gold(units=units, sid=name, session=session or name); g['dataset'] = 'synthetic'
    g['annotator'] = dict(tool='synthetic', model='fake-gold', prompt_version='synthetic-v1')
    write_json(s / 'gold.json', g)
    values = precision_values if precision_values is not None else [precision()]
    r = run(*[entry() for _ in values]); r['system'] = metadata or dict(name=system, llm='fake-product', parameters=dict(temperature=0))
    path = s / f'runs/{system}.json'; write_json(path, r)
    judgments = verdicts if verdicts is not None else [recall(u['unit_id']) for u in g['U']]
    j = dict(recall=judgments, precision=values, judge=dict(tool='synthetic', model='fake-judge', prompt_version='split-v1'), **input_hashes(s, path))
    write_json(s / f'judgments/{system}.json', j)
    return dict(gold=g, judgment=j, memories=[dict(memory_id=f'm{i}', statement=e['draft']['statement']) for i, e in enumerate(r['D'], 1)], sample_id=name)
