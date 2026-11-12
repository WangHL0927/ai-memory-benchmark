#!/usr/bin/env python3
"""发布前扫描与版本化导出（仅使用标准库）。

运行：python3 -m tools.publish scan/export --help。
扫描只写 tmp/publish/<清单名>/scan.{json,md}，不改输入。
位置使用 JSON Pointer（转义解码后的字符串）与零起始字符偏移；行、列从 1 开始。
allowlist.json 格式：{"entries": [{"id": "扫描命中 id", "reason": "误报理由"}]}。
redactions.json 格式：{"entries": [{"id": "扫描命中 id", "replacement": "[EMAIL_1]"}]}。
也可用 {"sample": "s001", "value": "原值", "replacement": "[EMAIL_1]"} 指定替换。
相同样本内同一原值只允许一种占位符，替换应用于全部字符串（包括冻结快照）。
扫描是启发式检查，不能证明没有个人信息；豁免需由操作者逐项判断。
"""
import argparse
from collections import Counter
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import unquote, urlsplit

from tools import report_split_html, score
from tools.model.judge_split import VERSIONS, memories, validate_judgment
from tools.config import sha256_file
from tools.score_split import identity_summary
from tools.paths import ROOT
from tools.trial import now, read_json, transcript


# 规则清单：依据为常见格式特征，偏向召回；不调用外部服务验证凭据是否有效。
RULES = [
    # 邮箱：RFC 5322 的常见 ASCII addr-spec 子集，不将无域名文本当邮箱。
    ('email', r'(?<![\w.+-])[\w.+-]+@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.[A-Za-z]{2,}(?![\w-]|\.[A-Za-z0-9])', '常见邮箱地址（RFC 5322 子集）'),
    # 电话：E.164 国际号、中国大陆手机号及常见北美分隔形式；排除连续版本/哈希中的数字。
    ('phone', r'(?<![\w.])(?:\+[1-9]\d{0,2}[ -]?(?:\(\d{1,4}\)|\d{1,4})(?:[ -]?\d){6,11}|1[3-9]\d{9}|\(?[2-9]\d{2}\)?[ .-]\d{3}[ .-]\d{4})(?![\w.]|[ -]?\d)', 'E.164、国内手机号、北美常见写法'),
    # IPv4：先找四段十进制候选，随后交 ipaddress 排除超范围的版本号。
    ('ipv4', r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?!\w|\.\d)', 'IPv4 点分十进制，ipaddress 复核'),
    # IPv6：覆盖压缩、完整、IPv4 映射和区域后缀，随后用标准库校验。
    ('ipv6', r'(?<![\w:])(?:[A-Fa-f0-9]*:){2,}[A-Fa-f0-9:.]*(?:%[\w.-]+)?(?![\w:])', 'IPv6 冒号格式，ipaddress 复核'),
    # URL：完整解析 userinfo 与查询参数，避免把普通 URL/文档链接全算敏感。
    ('url_credentials', r'\b[A-Za-z][A-Za-z0-9+.-]*://[^\s<>"\x27]+', 'RFC 3986 userinfo 或凭据查询参数'),
    # 前缀来自 OpenAI、GitHub、AWS、Slack 的公开凭据格式；长度下限过滤示例前缀。
    ('secret_token', r'(?<![\w-])(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|ASIA[A-Z0-9]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|AIza[A-Za-z0-9_-]{30,})(?![\w-])', 'OpenAI / GitHub / AWS / Slack / Google 的凭据前缀'),
    # 无固定前缀的凭据：常见配置键 + 至少八字符值；不把空值或占位符算密钥。
    ('secret_assignment', r'(?i)\b(?:api[_-]?key|client[_-]?secret|access[_-]?token|auth[_-]?token|password)\s*[:=]\s*["\x27]?([A-Za-z0-9_+/=-]{8,})', '常见配置中的凭据赋值，捕获值而非键名'),
    # JWT：RFC 7519 的三段 base64url 紧凑表示，头部需能解码且声明 alg。
    ('jwt', r'(?<![\w.-])eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+(?![\w.-])', 'JWT 三段紧凑格式与 JSON 头部'),
    # PEM/OpenSSH：RFC 7468 与 OpenSSH 的私钥标签，连同整个多行块处理。
    ('private_key', r'-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----[\s\S]*?-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----', 'PEM / OpenSSH 私钥块'),
    # 钱包：Ethereum 20 字节十六进制、Bitcoin Base58/Bech32、TRON Base58 的外观。
    ('wallet', r'(?<![A-Za-z0-9])(?:0x[A-Fa-f0-9]{40}|[13][a-km-zA-HJ-NP-Z1-9]{25,34}|(?:bc1|tb1)[ac-hj-np-z02-9]{11,71}|T[a-km-zA-HJ-NP-Z1-9]{33})(?![A-Za-z0-9])', 'Ethereum / Bitcoin / TRON 地址外观（未校验链上存在性）'),
    # 云资源：Vercel/OpenAI 等平台使用的项目、团队、组织前缀；裸前缀不命中。
    ('cloud_id', r'(?<![\w-])(?:prj_|proj_|team_|org_)[A-Za-z0-9_-]{6,}(?![\w-])|(?<![\w-])org-[A-Za-z0-9]{8,}(?![\w-])', '云平台 prj_ / proj_ / team_ / org_ / org- 标识'),
    # 路径：macOS、Linux、Windows 用户主目录；排除 /home/ 本身和变量占位符。
    ('user_path', r'(?:/Users/|/home/)[^/\s<>"\x27${}\[\]]+|(?i:[A-Z]:\\Users\\)[^/\\\s<>"\x27${}\[\]]+', '三类系统的用户主目录路径'),
]
COMPILED = [(kind, re.compile(pattern), basis) for kind, pattern, basis in RULES]
RULE_VERSION = hashlib.sha256(json.dumps(RULES).encode()).hexdigest()
TOKEN_QUERY = re.compile(r'(?i)(?:.*(?:token|secret|password|signature|credential|api[_-]?key).*|authorization|auth|key|sig|user(?:name)?|login|email)\Z')


def digest(content):
    return hashlib.sha256(content).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def safe_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError('路径标识只能含 ASCII 字母、数字、下划线、点和连字符')
    return value


def dataset_manifest(name):
    name = safe_name(name.removesuffix('.json'))
    path = ROOT / 'datasets' / (name + '.json')
    manifest = read_json(path)
    if manifest.get('redistribution') not in ('allowed', 'scores_only'):
        raise ValueError('清单必须明确声明 redistribution')
    for field in ('id', 'source', 'revision', 'license'):
        if not manifest.get(field):
            raise ValueError('数据集清单缺少 ' + field)
    if manifest['redistribution'] == 'allowed' and not manifest.get('attribution'):
        raise ValueError('允许再发布的数据集必须声明 attribution')
    return name, manifest, sha256_file(path)


def bounded(path):
    """发布输出禁止经符号链接落到别处，force 也不能扩大删除范围。"""
    path = Path(path)
    if path != ROOT and ROOT not in path.parents:
        raise ValueError('发布输出必须位于仓库内')
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('发布路径不能包含符号链接')
    return path


def inventory(data_dir):
    """一次读取原始字节；后续扫描、校验和导出均使用同一份快照。"""
    data_dir = Path(data_dir).resolve()
    if not data_dir.is_dir():
        raise ValueError('评测数据目录不存在')
    files = {}
    for sample in sorted(data_dir.iterdir()):
        if not sample.is_dir() or sample.name.startswith(('.', '_')):
            continue
        paths = [p for p in (sample / 'transcript.md', sample / 'gold.json') if p.exists()]
        paths += sorted((sample / 'runs').glob('*.json'))
        paths += sorted((sample / 'judgments').glob('*.json'))
        if not paths:
            continue
        safe_name(sample.name)
        for path in paths:
            if any(p.is_symlink() for p in (sample, path.parent, path)):
                raise ValueError('输入产物不能使用符号链接')
            safe_name(path.stem)
            files[path.relative_to(data_dir).as_posix()] = path.read_bytes()
    if not files:
        raise ValueError('没有找到 transcript、gold、runs 或 judgments 产物')
    return files


def leaves(value, pointer=''):
    """解码 JSON 后扫描，避免反斜杠、Unicode 转义和多行私钥逃过检查。"""
    if isinstance(value, str):
        yield pointer, value
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from leaves(child, pointer + '/' + str(i))
    elif isinstance(value, dict):
        for key, child in value.items():
            ptr = pointer + '/' + key.replace('~', '~0').replace('/', '~1')
            yield ptr + '/@key', key
            yield from leaves(child, ptr)


def detections(text):
    import base64
    for kind, pattern, _ in COMPILED:
        for match in pattern.finditer(text):
            start, end = match.span(1) if kind == 'secret_assignment' else match.span()
            value = text[start:end]
            if kind == 'ipv6':
                # 句末英文句号属于标点；IPv4 映射地址内部的点仍保留。
                value = value.rstrip('.')
                end = start + len(value)
            if kind in ('ipv4', 'ipv6'):
                try:
                    ipaddress.ip_address(value.split('%')[0])
                except ValueError:
                    continue
            if kind == 'jwt':
                try:
                    head = value.split('.')[0]
                    if not isinstance(json.loads(base64.urlsafe_b64decode(head + '=' * (-len(head) % 4))), dict):
                        continue
                    if 'alg' not in json.loads(base64.urlsafe_b64decode(head + '=' * (-len(head) % 4))):
                        continue
                except (ValueError, UnicodeError):
                    continue
            if kind == 'url_credentials':
                try:
                    parsed = urlsplit(value)
                except ValueError:
                    continue
                # 只捕获敏感分量，确保单独引用 token 的证据与完整 URL 使用同一替换值。
                authority = re.match(r'[^:]+://([^/?#]+)', value)
                if authority and '@' in authority[1]:
                    userinfo = authority[1].rsplit('@', 1)[0]
                    offset = authority.start(1)
                    for component in userinfo.split(':', 1):
                        if component:
                            yield kind, start + offset, start + offset + len(component)
                        offset += len(component) + 1
                for param in re.finditer(r'[?&#;]([^=?&#;]+)=([^&#;]*)', value):
                    if (param[2] and TOKEN_QUERY.fullmatch(unquote(param[1]))
                            and not re.fullmatch(r'\[[A-Z][A-Z0-9_]*\]', param[2])):
                        yield kind, start + param.start(2), start + param.end(2)
                continue
            yield kind, start, end


def scan_files(files):
    hits = []
    for filename, content in sorted(files.items()):
        text = content.decode('utf-8')
        values = leaves(json.loads(text)) if filename.endswith('.json') else [('', text)]
        for pointer, value in values:
            found = list(detections(value))
            # JSON 把配置键和值拆开保存；不能只在字符串内部找「api_key=...」。
            key = pointer.rsplit('/', 1)[-1].replace('~1', '/').replace('~0', '~')
            if (re.fullmatch(r'(?i)(?:api[_-]?key|client[_-]?secret|access[_-]?token|auth[_-]?token|password)', key)
                    and re.fullmatch(r'[A-Za-z0-9_+/=-]{8,}', value)):
                found.append(('secret_assignment', 0, len(value)))
            for kind, start, end in sorted(set(found)):
                location = dict(pointer=pointer, start=start, end=end,
                                line=value.count('\n', 0, start) + 1,
                                column=start - value.rfind('\n', 0, start))
                hit = dict(sample=filename.split('/')[0], file=filename, location=location,
                           type=kind, value=value[start:end], before=value[max(0, start-30):start],
                           after=value[end:end+30])
                hit['id'] = digest(json_bytes(hit))[:24]
                hits.append(hit)
    return hits


def entries(path):
    if not path.exists():
        return []
    value = read_json(path)
    result = value.get('entries') if isinstance(value, dict) else value
    if not isinstance(result, list) or any(not isinstance(e, dict) for e in result):
        raise ValueError(path.name + ' 必须是 entries 数组')
    return result


def exemptions(folder, hits):
    known = {h['id'] for h in hits}
    allowed = set()
    for item in entries(folder / 'allowlist.json'):
        if not isinstance(item.get('id'), str) or not isinstance(item.get('reason'), str) or not item['reason'].strip():
            raise ValueError('allowlist 每项必须给出命中 id 和非空 reason，禁止泛化豁免')
        if item['id'] in known:
            allowed.add(item['id'])
    return allowed


def scan(data_dir, dataset):
    name, meta, meta_hash = dataset_manifest(dataset)
    folder = bounded(ROOT / 'tmp/publish' / name)
    # 扫描报告若预先被换成输入文件的符号链接，也必须保持只读承诺。
    for filename in ('scan.json', 'scan.md'):
        bounded(folder / filename)
    files = inventory(data_dir)
    hits = scan_files(files)
    allowed = exemptions(folder, hits)
    result = dict(schema_version=1, dataset=name, dataset_sha256=meta_hash, rule_version=RULE_VERSION,
                  scanned_at=now(), files={p: digest(b) for p, b in files.items()}, hits=hits)
    lines = ['# 发布前敏感信息扫描', '', '仅检查，不修改输入；JSON 位置基于解码字符串。前后各最多 30 字符。', '',
             '规则清单：'] + [f'- `{kind}`：{basis}' for kind, _, basis in RULES]
    lines += ['', f'命中 {len(hits)} 条；当前豁免 {len(allowed)} 条。', '']
    for hit in hits:
        lines += [f"- {hit['id']} · {hit['sample']} · {hit['file']} · {hit['type']} · {'已豁免' if hit['id'] in allowed else '待处理'}",
                  '  位置：' + json.dumps(hit['location'], ensure_ascii=False),
                  '  上下文：' + json.dumps(hit['before'] + '【' + hit['value'] + '】' + hit['after'], ensure_ascii=False)]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'scan.json').write_bytes(json_bytes(result))
    (folder / 'scan.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


def replacement_maps(folder, hits, allowed):
    known = {h['id']: h for h in hits}
    maps = {}
    for item in entries(folder / 'redactions.json'):
        if 'id' in item:
            if item['id'] not in known:
                raise ValueError('redactions 含失效的命中 id，请重新核对扫描')
            sample, value = known[item['id']]['sample'], known[item['id']]['value']
        else:
            sample, value = item.get('sample'), item.get('value')
            if not any(h['sample'] == sample and h['value'] == value for h in hits):
                raise ValueError('redactions 必须引用本轮扫描命中')
        placeholder = item.get('replacement')
        if not isinstance(placeholder, str) or not re.fullmatch(r'\[[A-Z][A-Z0-9_]*\]', placeholder):
            raise ValueError('replacement 必须是 [EMAIL_1] 形式的占位符')
        mapping = maps.setdefault(sample, {})
        if value in mapping and mapping[value] != placeholder:
            raise ValueError('同一样本同一原值出现冲突占位符')
        if placeholder in mapping.values() and mapping.get(value) != placeholder:
            raise ValueError('同一样本的不同原值不能共用占位符')
        mapping[value] = placeholder
    pending = [h for h in hits if h['id'] not in allowed and h['value'] not in maps.get(h['sample'], {})]
    if pending:
        # 只列位置和类型，避免终端泄露扫描原文。
        detail = '\n'.join(f"{h['id']} {h['file']} {h['location']['pointer']}:{h['location']['start']} {h['type']}" for h in pending)
        raise ValueError(f'存在 {len(pending)} 条未处理命中：\n' + detail)
    return maps


def replace_text(text, mapping):
    """单次最长优先替换，避免邮箱内 token 或嵌套 URL 被分步替坏。"""
    if not mapping:
        return text
    pattern = re.compile('|'.join(re.escape(v) for v in sorted(mapping, key=len, reverse=True)))
    return pattern.sub(lambda m: mapping[m[0]], text)


def transform(value, mapping):
    if isinstance(value, str):
        return replace_text(value, mapping)
    if isinstance(value, list):
        return [transform(v, mapping) for v in value]
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            key = replace_text(key, mapping)
            if key in result:
                raise ValueError('脱敏造成 JSON 键冲突')
            result[key] = transform(child, mapping)
        return result
    return value


def check_evidence(value, turns):
    """递归检查 gold、冻结快照和 runs 的所有 evidence，要求对应轮次连续原文。"""
    if isinstance(value, list):
        for child in value:
            check_evidence(child, turns)
    elif isinstance(value, dict):
        if 'evidence' in value:
            if not isinstance(value['evidence'], list):
                raise ValueError('evidence 必须是数组')
            for item in value['evidence']:
                if (not isinstance(item, dict) or type(item.get('turn')) is not int
                        or not isinstance(item.get('quote'), str) or not item['quote']
                        or item['quote'] not in turns.get(item['turn'], '')):
                    raise ValueError('脱敏后证据无法在对应 transcript 轮次中找到')
        for child in value.values():
            check_evidence(child, turns)


def checked_data(files, maps, policy):
    # 先校验未加工判定，不能通过脱敏重绑哈希把过期判定变成有效判定。
    for filename, content in files.items():
        if '/judgments/' in filename:
            sample, _, name = filename.split('/')
            run_path = sample + '/runs/' + name
            hashes = dict(run_sha256=digest(files[run_path]), gold_sha256=digest(files[sample + '/gold.json']),
                          transcript_sha256=digest(files[sample + '/transcript.md']))
            validate_judgment(json.loads(content), json.loads(files[sample + '/gold.json']),
                              memories(json.loads(files[run_path])), hashes)
    result = {}
    for filename, content in files.items():
        mapping = maps.get(filename.split('/')[0], {})
        text = content.decode('utf-8')
        result[filename] = json_bytes(transform(json.loads(text), mapping)) if filename.endswith('.json') else replace_text(text, mapping).encode('utf-8')
    for sample in sorted({p.split('/')[0] for p in result}):
        if sample + '/gold.json' not in result or sample + '/transcript.md' not in result:
            raise ValueError('样本必须同时具有 transcript.md 和 gold.json')
        gold = json.loads(result[sample + '/gold.json'])
        score.validate_gold(gold, policy)
        if gold['sample_id'] != sample:
            raise ValueError('sample_id 与样本目录不一致')
        # 本地会话即使被错误放进公共数据集目录，也不能借清单取得发布资格。
        if gold.get('source') in ('claude', 'claude-code', 'codex', 'private', 'local'):
            raise ValueError('禁止公开自有或私有会话')
        _, parsed = transcript(result[sample + '/transcript.md'].decode('utf-8'))
        turns = {t['turn']: t['text'] for t in parsed}
        check_evidence(gold, turns)
        for filename, content in result.items():
            if filename.startswith(sample + '/runs/'):
                run = json.loads(content)
                if run.get('policy') != gold['policy'] or run.get('standard_version') != gold['standard_version']:
                    raise ValueError('run 与 gold 的政策或框架版本不一致')
                check_evidence(run, turns)
    for filename in list(result):
        if '/judgments/' in filename:
            sample, _, name = filename.split('/')
            run_path = sample + '/runs/' + name
            hashes = dict(run_sha256=digest(result[run_path]), gold_sha256=digest(result[sample + '/gold.json']),
                          transcript_sha256=digest(result[sample + '/transcript.md']))
            judgment = dict(json.loads(result[filename]), **hashes)
            validate_judgment(judgment, json.loads(result[sample + '/gold.json']),
                              memories(json.loads(result[run_path])), hashes)
            result[filename] = json_bytes(judgment)
    return result


def aggregate_only(value):
    """保留分数和统计结构，去掉逐条原文、标识明细及输入绝对路径。"""
    if isinstance(value, dict):
        return {k: aggregate_only(v) for k, v in value.items()
                if k not in ('details', 'files', 'data_dir', 'evidence', 'quote', 'statement', 'draft', 'note', 'valid_samples', 'failures', 'sample_id', 'session_id')
                and not (k == 'samples' and isinstance(v, (list, dict)))
                and not (k in ('errors', 'raw') and isinstance(v, (str, list, dict)))}
    if isinstance(value, list):
        return [aggregate_only(v) for v in value]
    return value


def public_report(report, maps, meta):
    # 顶层白名单防止调用方把整个 transcript 或任意附加数据夹入 report.json。
    keys = ('protocol', 'precision_version', 'false_memory_unit', 'system_summary', 'sample_count', 'bootstrap', 'warnings', 'datasets', 'agreement',
            'annotators', 'judges', 'systems', 'calibration_notice', 'native_differences',
            'failure_count', 'failure_reason_counts')
    clean = {k: aggregate_only(report[k]) for k in keys if k in report}
    clean['failures'] = []  # 发布汇总数，不发布逐样本错误明细。
    mapping = {}
    for replacements in maps.values():
        for value, placeholder in replacements.items():
            if value in mapping and mapping[value] != placeholder:
                # 样本内的一致性不要求跨样本共用编号；汇总元信息使用独立占位符。
                mapping[value] = '[REPORT_REDACTED_' + digest(value.encode())[:8].upper() + ']'
            else:
                mapping[value] = placeholder
    clean = transform(clean, mapping)
    clean['publication'] = dict(dataset=meta['id'], redistribution=meta['redistribution'],
                               details_omitted=True, calibration='已附一致率，门槛未确认' if clean.get('agreement') else '未经人工校准')
    return clean


def verify_report(report, files):
    """拆分报告的输入哈希、数据集及框架/政策声明必须对应当前快照。"""
    golds = {p for p in files if p.endswith('/gold.json')}
    if (report.get('protocol') != 'split-v1' or report.get('precision_version') != VERSIONS['precision']
            or report.get('false_memory_unit') != 'claim' or report.get('sample_count') != len(golds)):
        raise ValueError('拆分报告协议、精确判定版本、计数口径或样本数与数据目录不一致')
    references = report.get('files')
    if not isinstance(references, list) or not references:
        raise ValueError('报告缺少评分输入 files 哈希清单')
    seen = set()
    for item in references:
        path = Path(item['path'])
        relative = '/'.join(path.parts[-3:]) if path.parent.name in ('runs', 'judgments') else '/'.join(path.parts[-2:])
        if relative not in files or item.get('sha256') != digest(files[relative]):
            raise ValueError('报告输入哈希与当前数据不一致')
        seen.add(relative)
    if not golds <= seen or not any('/runs/' in p for p in seen):
        raise ValueError('报告未覆盖当前全部样本或没有被测输出')
    for path in seen:
        if '/runs/' in path:
            sample, _, name = path.split('/')
            if not {sample + '/transcript.md', sample + '/judgments/' + name} <= seen:
                raise ValueError('报告缺少拆分判定或 transcript 输入哈希')
    datasets = {}
    for path in golds:
        gold = json.loads(files[path])
        dataset = str(gold.get('dataset') or gold.get('source') or Path(report['data_dir']).name)
        group = datasets.setdefault(dataset, dict(samples=set(), protocols=set()))
        group['samples'].add(path.split('/')[0])
        group['protocols'].add(json.dumps([gold['standard_version'], gold['policy']], sort_keys=True))
        score.validate_gold(gold)
    if set(report.get('datasets', {})) != set(datasets):
        raise ValueError('报告数据集与 gold 不一致')
    for dataset, expected in datasets.items():
        table = report['datasets'][dataset]
        if (table.get('sample_count') != len(expected['samples']) or set(table.get('samples', [])) != expected['samples']
                or {json.dumps(p, sort_keys=True) for p in table.get('protocol_versions', [])} != expected['protocols']):
            raise ValueError('报告与 gold 的数据集、政策或框架版本不一致')


def provenance(meta, report, version, redaction_count, allow_count):
    def show(value):
        return '未记录' if value is None or value == [None] or value == [] else json.dumps(value, ensure_ascii=False, sort_keys=True)
    lines = ['# 发布来源与加工说明', '', f'评测版本：{version}', f'导出时间（UTC）：{now()}',
             f"数据集：{meta['id']}", f"来源：{meta['source']}", f"固定 revision：{meta['revision']}",
             f"许可：{meta['license']}", '署名：' + meta.get('attribution', '未记录（本批不再发布数据）'),
             f"再发布方式：{meta['redistribution']}", '政策版本：' + show({d: t['protocol_versions'] for d, t in report['datasets'].items()}),
             '框架版本：' + show(sorted({v[0] for t in report['datasets'].values() for v in t['protocol_versions']})), '标注者身份与模型参数：' + show(identity_summary(report.get('annotators', []))),
             '判定者身份与模型参数：' + show(identity_summary(report.get('judges', []))), '被测对象与模型参数：' + show(report.get('system_summary')),
             '样本数：' + str(report['sample_count']), '精确判定：' + report['precision_version'] + '；误记率按陈述计，按条口径为辅助指标。', '',
             f'脱敏处理：按样本一致替换 {redaction_count} 个原值；逐项豁免 {allow_count} 条命中。' if meta['redistribution'] == 'allowed' else '脱敏处理：不发布 transcript、gold、runs、judgments，仅发布汇总分数、元信息和输入产物哈希。',
             '报告移除了逐项错误、原文和本机输入路径。manifest 的文件路径相对仓库根目录；manifest 自身不做自引用哈希。']
    lines += ['一致率：' + show(report['agreement']) if report.get('agreement') else '未经人工校准：报告未提供人工一致率，仅供试跑，不能视为已校准成绩。']
    return '\n\n'.join(lines) + '\n'


def install(trees, force):
    """先完成全部校验及暂存，再发布目录；失败时恢复 force 替换前的目录。"""
    scratch = bounded(ROOT / 'tmp/publish')
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.export-', dir=scratch) as temporary:
        temporary = Path(temporary)
        staged, backups, installed = {}, {}, []
        for i, (target, files) in enumerate(trees.items()):
            bounded(target)
            if target.exists() and not force:
                raise ValueError('导出目标已存在，请更换版本或使用 --force')
            stage = temporary / ('stage-' + str(i))
            stage.mkdir()
            for relative, content in files.items():
                path = stage / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            staged[target] = stage
        try:
            for i, (target, stage) in enumerate(staged.items()):
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if not force:
                        raise ValueError('导出目标已存在')
                    backup = temporary / ('backup-' + str(i))
                    os.rename(target, backup)
                    backups[target] = backup
                os.rename(stage, target)
                installed.append(target)
        except BaseException:
            for target in reversed(installed):
                shutil.rmtree(target)
            for target, backup in backups.items():
                os.rename(backup, target)
            raise


def export(data_dir, dataset, eval_version, report_path, force=False):
    name, meta, meta_hash = dataset_manifest(dataset)  # 许可检查必须先于任何公开写入。
    version = safe_name(eval_version)
    files = inventory(data_dir)
    report = read_json(report_path)
    verify_report(report, files)
    policies = {json.dumps(json.loads(b)['policy'], sort_keys=True) for p, b in files.items() if p.endswith('/gold.json')}
    if len(policies) != 1:
        raise ValueError('一次发布的数据必须使用同一份政策')
    reference = json.loads(policies.pop())
    policy = score.load_policy(reference=reference)
    folder = bounded(ROOT / 'tmp/publish' / name)
    maps, allowed, hits, output_data = {}, set(), [], {}
    if meta['redistribution'] == 'allowed':
        snapshot_path = folder / 'scan.json'
        if not snapshot_path.exists():
            raise ValueError('允许再发布的数据集必须先运行 scan')
        snapshot = read_json(snapshot_path)
        hits = scan_files(files)
        if (snapshot.get('dataset') != name or snapshot.get('dataset_sha256') != meta_hash
                or snapshot.get('rule_version') != RULE_VERSION
                or snapshot.get('files') != {p: digest(b) for p, b in files.items()}
                or snapshot.get('hits') != hits):
            raise ValueError('scan.json 已过期或被修改，请重新运行 scan')
        allowed = exemptions(folder, hits)
        maps = replacement_maps(folder, hits, allowed)
        output_data = checked_data(files, maps, policy)
        # 最长匹配替换可能遮住相交命中：对最终字节重扫，不能遗留未豁免秘密。
        permitted = {(h['sample'], h['type'], h['value']) for h in hits if h['id'] in allowed}
        if any((h['sample'], h['type'], h['value']) not in permitted for h in scan_files(output_data)):
            raise ValueError('脱敏后仍有未豁免命中，请核对相交替换和占位符')
    clean = public_report(report, maps, meta)
    # 报告元信息也可能含凭据，不能通过 HTML/PROVENANCE 绕过数据扫描。
    allowed_values = {(h['type'], h['value']) for h in hits if h['id'] in allowed}
    if any((h['type'], h['value']) not in allowed_values for h in scan_files({'report/report.json': json_bytes(clean)})):
        raise ValueError('报告元信息仍含敏感标识，请先修正报告或补充脱敏处理')
    reports = bounded(ROOT / 'reports' / version)
    data = bounded(ROOT / 'data' / name / (safe_name(reference['id']) + '-' + safe_name(reference['version'])))
    # 禁止 force 的目标与输入祖先/子孙重合，避免替换时删除评测源文件。
    source = Path(data_dir).resolve()
    targets = [reports] + ([data] if output_data else [])
    if any(target == source or target in source.parents or source in target.parents for target in targets):
        raise ValueError('导出目录不能与输入数据重叠')
    report_files = {'report.json': json_bytes(clean), 'report.html': report_split_html.render(clean).encode('utf-8'),
                    'PROVENANCE.md': provenance(meta, clean, version, sum(map(len, maps.values())), len(allowed)).encode('utf-8')}
    hashes = {str((reports / p).relative_to(ROOT)): digest(b) for p, b in report_files.items()}
    hashes.update({str((data / p).relative_to(ROOT)): digest(b) for p, b in output_data.items()})
    manifest = dict(schema_version=1, dataset=name, dataset_revision=meta['revision'], dataset_manifest_sha256=meta_hash,
                    eval_version=version, policy=reference, redistribution=meta['redistribution'],
                    files=hashes, artifacts={p: digest(b) for p, b in files.items()})
    report_files['manifest.json'] = json_bytes(manifest)
    trees = {reports: report_files}
    if output_data:
        trees[data] = output_data
    install(trees, force)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('scan', 'export'):
        sub = commands.add_parser(command)
        sub.add_argument('--data-dir', type=Path, required=True)
        sub.add_argument('--dataset', required=True, help='datasets/ 下的清单名，可带 .json 后缀')
        if command == 'export':
            sub.add_argument('--eval-version', required=True)
            sub.add_argument('--report', type=Path, required=True)
            sub.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'scan':
            result = scan(args.data_dir, args.dataset)
            print('扫描完成：' + json.dumps(dict(Counter(h['type'] for h in result['hits'])), ensure_ascii=False) + '；结果位于 tmp/publish/' + result['dataset'])
        else:
            result = export(args.data_dir, args.dataset, args.eval_version, args.report, args.force)
            print(f"导出完成：reports/{args.eval_version}；公开数据文件 {sum(p.startswith('data/') for p in result['files'])} 个")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f'错误：{exc}\n')


if __name__ == '__main__':
    main()
