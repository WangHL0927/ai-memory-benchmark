"""只给没有 gold 的样本盲标；校验后原子写入冻结金标，不读取 runs。"""
import copy
import re
import json
from tools import score
from tools.paths import TMP
from tools.model.common import arguments, cli_json, prompt_file
from tools.trial import batch, now, output_path, policy_material, section, transcript, write_json

PROMPT_VERSION = 'model-annotate-v1'


def _squash(value):
    return re.sub(r'\s+', ' ', value).strip()


def _unescape(value):
    return value.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')


def repair_units(units, turns, log):
    """确定性修复，不调用模型。证据比对：空白合并后逐字一致；每轮比两个版本：原文，以及把工具参数里的
    JSON 转义还原后的原文（transcript 里工具参数按 JSON 显示，模型常按还原后的代码引用）。
    引文只在另一轮出现一次就改正轮次；找不到的证据删除；没有证据剩下的单元删除。计数写进 log。"""
    variants = {t['turn']: (_squash(t['text']), _squash(_unescape(t['text']))) for t in turns}
    kept = []
    for unit in units:
        evidence = []
        for item in unit.get('evidence') or []:
            quote = _squash(item['quote']) if isinstance(item.get('quote'), str) else ''
            if quote and any(quote in v for v in variants.get(item.get('turn'), ())):
                evidence.append(item)
                continue
            hits = [n for n, vs in variants.items() if quote and any(quote in v for v in vs)]
            if len(hits) == 1:
                log['turn_fixed'] += 1
                evidence.append(dict(item, turn=hits[0]))
            else:
                log['evidence_dropped'] += 1
        log['evidence_total'] += len(unit.get('evidence') or [])
        if evidence:
            kept.append(dict(unit, evidence=evidence))
        else:
            log['units_dropped'].append(unit.get('unit_id'))
    return kept


def annotate(sample, args):
    target = output_path(sample / 'gold.json', args.data_dir)
    if target.exists():
        return 'skipped'
    text = (sample / 'transcript.md').read_text(encoding='utf-8')
    meta, turns = transcript(text)
    policy, md = policy_material(args.policy, {'id': 'default', 'version': 'v1'})
    version = 'v0.4'  # 数据格式版本；框架文档的精确版本写在运行环境里
    template = prompt_file('annotate')
    prompt = '\n\n'.join([template, section('4'), section('5'), md,
                           '政策 JSON：' + json.dumps(policy, ensure_ascii=False), text])
    def validate(value):
        if not isinstance(value, dict) or not all(k in value for k in ('project', 'U', 'N')):
            raise ValueError('盲标缺 project/U/N')
        log = dict(evidence_total=0, turn_fixed=0, evidence_dropped=0, units_dropped=[])
        value = dict(value, U=repair_units(value['U'], turns, log), N=repair_units(value['N'], turns, log))
        if log['evidence_total'] and log['evidence_dropped'] > 0.2 * log['evidence_total']:
            raise ValueError(f"证据对不上原文的比例过高：{log['evidence_dropped']}/{log['evidence_total']}")
        gold = dict(sample_id=sample.name, session_id=meta.get('会话 id') or sample.name,
                    source=meta.get('来源') or 'unknown', turns=max(t['turn'] for t in turns),
                    dataset=meta.get('数据集', meta.get('来源', 'unknown')), layer=args.layer,
                    project=value['project'], standard_version=version, policy=score.policy_ref(policy),
                    frozen_at=now(), gold_revision=0, U=value['U'], N=value['N'])
        gold.update(frozen_policy=copy.deepcopy(gold['policy']),
                    frozen_project=copy.deepcopy(gold['project']), blind_snapshot=copy.deepcopy(dict(U=gold['U'], N=gold['N'])))
        gold['annotation_repairs'] = log
        score.validate_gold(gold, policy)
        for unit in gold['U'] + gold['N']:
            if unit['origin'] != 'blind' or score.scope_error(unit['scope'], policy, gold['project']):
                raise ValueError('盲标 origin/scope 无效')
        return gold
    gold, identity = cli_json(args, prompt, PROMPT_VERSION, validate, stage='annotate', template=template)
    gold['annotator'] = identity
    if getattr(args, 'dataset', None):
        gold['dataset'] = args.dataset
    write_json(target, gold, overwrite=False)
    return 'written'


def main():
    parser = arguments(__doc__)
    parser.add_argument('--dataset', help='数据集标识，覆盖 transcript 中的来源')
    parser.add_argument('--layer', choices=['random', 'targeted'], default='random')
    args = parser.parse_args()
    try:
        return batch(sorted(p.parent for p in args.data_dir.glob('*/transcript.md')),
                     lambda p: annotate(p, args), args.failures or TMP / 'trial/annotate-failures.json', args.concurrency)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'错误：{exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
