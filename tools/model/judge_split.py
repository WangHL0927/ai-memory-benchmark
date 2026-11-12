"""召回 / 精确两种模型小判断；独立落盘，精确判断按陈述计数。"""
import json
from pathlib import Path

from tools.config import sha256_file
from tools import score
from tools.model.common import arguments, cli_json, prompt_file
from tools.paths import TMP
from tools.trial import batch, now, output_path, policy_material, read_json, write_json

VERSIONS = {'recall': 'model-judge-recall-v1', 'precision': 'model-judge-precision-v4'}
BATCH_SIZE = 30
CLAIM_FIELDS = ('claims', 'unsupported', 'partial', 'distorted_claims')


def input_hashes(sample, run_path):
    return dict(run_sha256=sha256_file(run_path), gold_sha256=sha256_file(sample / 'gold.json'),
                transcript_sha256=sha256_file(sample / 'transcript.md'))


def memories(run):
    """稳定的 m 编号只取决于原始 D 顺序，不向模型提供产品元信息。"""
    values = []
    for i, entry in enumerate(score.items(run), 1):
        score.require(isinstance(entry, dict), '提交项必须为对象')
        draft = entry.get('draft', entry)
        score.require(isinstance(draft, dict) and isinstance(draft.get('statement'), str)
                      and bool(draft['statement'].strip()), f'm{i} 缺少非空 statement')
        values.append(dict(memory_id=f'm{i}', statement=draft['statement']))
    return values


def validate_claim_counts(value):
    """模型判定和独立人工计数共用；bool 不是合法整数计数。"""
    for field in CLAIM_FIELDS:
        score.require(type(value.get(field)) is int and value[field] >= 0, field + ' 必须为非负整数')
    score.require(value['claims'] >= sum(value[k] for k in CLAIM_FIELDS[1:]), '陈述分类计数之和超过 claims；每个陈述只能归入一类')
    score.require(type(value.get('verbatim')) is bool, 'verbatim 必须为布尔值')
    score.require(not value['verbatim'] or value['claims'] >= 1, '含原文摘录时 claims 至少为 1')


def validate_batch(values, ids, kind, memory_ids=()):
    """按编号校验双射并恢复输入顺序，不能用 zip 静默吞掉缺项或错号。"""
    key = 'unit_id' if kind == 'recall' else 'memory_id'
    score.require(kind in VERSIONS and isinstance(values, list) and len(values) == len(ids), '批次判定数量不符')
    found = {}
    fields = {key, 'verdict', 'memories', 'note'} if kind == 'recall' else {key, 'grounded', 'distorted', 'worth', 'note', 'verbatim', 'errors', *CLAIM_FIELDS}
    for value in values:
        score.require(isinstance(value, dict) and set(value) == fields, '批次判定字段不符')
        identifier = value[key]
        score.require(isinstance(identifier, str) and identifier in ids and identifier not in found, '批次编号缺失、重复或多出')
        score.require(isinstance(value['note'], str), 'note 必须为文本')
        if kind == 'recall':
            score.require(isinstance(value['verdict'], str) and value['verdict'] in ('full', 'partial', 'none', 'distorted'), '召回 verdict 无效')
            refs = value['memories']
            score.require(isinstance(refs, list) and all(isinstance(m, str) and m in memory_ids for m in refs)
                          and len(set(refs)) == len(refs), '支撑记忆编号无效')
            score.require(value['verdict'] not in ('full', 'partial') or bool(refs), '覆盖判定缺少支撑记忆')
            score.require(value['verdict'] != 'none' or not refs, 'none 不应含支撑记忆')
        else:
            score.require(isinstance(value['grounded'], str) and value['grounded'] in ('yes', 'partial', 'no'), 'grounded 无效')
            score.require(type(value['distorted']) is bool, 'distorted 必须为布尔值')
            score.require(isinstance(value['worth'], str) and value['worth'] in ('yes', 'no', 'unclear'), 'worth 无效')
            validate_claim_counts(value)
            score.require(isinstance(value['errors'], list), 'errors 必须为数组')
            # ponytail: 只是示例列表，超过 10 项时截取前 10 项，计数字段不受影响。
            errors = value['errors'] = value['errors'][:10]
            error_counts = dict(unsupported=0, distorted=0)
            for error in errors:
                score.require(isinstance(error, dict) and set(error) == {'text', 'type', 'reason'}, 'errors 项字段不符')
                score.require(isinstance(error['type'], str) and error['type'] in error_counts, 'errors.type 无效')
                score.require(all(isinstance(error[k], str) and error[k].strip() for k in ('text', 'reason')), 'errors 引用和理由必须为非空文本')
                error_counts[error['type']] += 1
            score.require(error_counts['unsupported'] <= value['unsupported'] and
                          error_counts['distorted'] <= value['distorted_claims'], 'errors 分类数量超过对应陈述计数')
        found[identifier] = value
    return [found[i] for i in ids]


def validate_judgment(value, gold, submitted, hashes):
    score.require(isinstance(value, dict) and all(value.get(k) == v for k, v in hashes.items()), '判定已过期：输入 sha256 不匹配')
    score.require(isinstance(value.get('judge'), dict) and bool(value['judge']), '判定缺少身份记录')
    mids = [m['memory_id'] for m in submitted]
    recall = validate_batch(value.get('recall'), [u['unit_id'] for u in gold['U']], 'recall', mids)
    precision = validate_batch(value.get('precision'), mids, 'precision')
    return dict(value, recall=recall, precision=precision)


def run_failure(run):
    """失败产物不能冒充空记忆的成功回放。"""
    status = run.get('status')
    if status is not None and status not in ('written', 'success', 'completed', 'ok'):
        return str(run.get('reason') or run.get('error') or status)
    system = run.get('system') or {}
    if isinstance(system, dict) and isinstance(system.get('timeout'), dict) and system['timeout'].get('occurred') is True:
        return '回放超时'
    return None


def judge(path, args):
    path = Path(path)
    sample = path.parent.parent
    target = output_path(sample / 'judgments' / path.name, args.data_dir)
    hashes = input_hashes(sample, path)
    gold, run = read_json(sample / 'gold.json'), read_json(path)
    score.require(isinstance(run, dict), 'run 必须为对象')
    score.require(not run_failure(run), '无法回放：' + str(run_failure(run)))
    submitted = memories(run)
    if target.exists() and not args.force:
        try:
            validate_judgment(read_json(target), gold, submitted, hashes)
            return 'skipped'
        except (ValueError, KeyError, TypeError):
            pass  # 过期或损坏的产物不算可复用判定；写成功之前仍保留原文件。
    policy, md = policy_material(args.policy, gold.get('policy'))
    score.validate_gold(gold, policy)
    score.require(run.get('standard_version') == gold['standard_version'] and run.get('policy') == gold['policy'], 'run 的框架或政策不匹配')
    llm = (run.get('system') or {}).get('llm')
    if isinstance(llm, dict):
        llm = llm.get('model') or llm.get('name')
    score.require(not llm or llm != args.model, '判定模型不能与被测模型相同')
    units = [{k: u[k] for k in ('unit_id', 'statement', 'conditions', 'category')} for u in gold['U']]
    text = (sample / 'transcript.md').read_text(encoding='utf-8')
    result, identities = dict(recall=[], precision=[]), []
    mids = [m['memory_id'] for m in submitted]
    for kind, values in (('recall', units), ('precision', submitted)):
        template = prompt_file('judge-' + kind)
        size = getattr(args, 'batch_size', None) or BATCH_SIZE
        for offset in range(0, len(values), size):
            part = values[offset:offset + size]
            key = 'unit_id' if kind == 'recall' else 'memory_id'
            ids = [v[key] for v in part]
            material = (dict(units=part, memories=submitted) if kind == 'recall' else
                        dict(transcript=text, memories=part, policy=policy, policy_text=md))
            prompt = template + '\n\n输入材料 JSON：\n' + json.dumps(material, ensure_ascii=False)
            verdicts, identity = cli_json(args, prompt, VERSIONS[kind],
                lambda v: validate_batch(v, ids, kind, mids), stage='judge', template=template)
            result[kind].extend(verdicts)
            identities.append(dict(kind=kind, batch=offset // size + 1, ids=ids, identity=identity))
    result.update(hashes)
    result['judge'] = dict(tool=args.annotator, model=args.model, reasoning_effort=args.reasoning_effort,
                          prompt_version='model-judge-split-v1', prompt_versions=VERSIONS,
                          protocol='split-v1', batches=identities, time=now())
    validate_judgment(result, gold, submitted, hashes)
    score.require(input_hashes(sample, path) == hashes, '等待期间 gold/run/transcript 已修改，拒绝写入')
    write_json(target, result)
    return 'written'


def main():
    parser = arguments(__doc__)
    parser.add_argument('--systems', nargs='+')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--batch-size', type=int, help=f'每次判定调用的记忆 / 单元数，默认 {BATCH_SIZE}；记忆很长时调小')
    args = parser.parse_args()
    try:
        tasks = [p for p in sorted(args.data_dir.glob('*/runs/*.json')) if not args.systems or p.stem in args.systems]
        return batch(tasks, lambda p: judge(p, args), args.failures or TMP / 'trial/judge-split-failures.json', args.concurrency)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'错误：{exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
