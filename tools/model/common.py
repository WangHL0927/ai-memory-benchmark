"""CLI 调用协议：独立临时工作目录、JSON 输出、一次重试。"""
import argparse
import copy
from pathlib import Path
import subprocess
import tempfile
from tools.config import cli_version, environment, sha256_text
from tools.cache import ModelCache, save_raw
from tools.paths import ROOT, TMP
from tools.trial import ModelOutputError, now, parse_json


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--annotator', required=True, choices=['codex', 'claude'])
    parser.add_argument('--model', required=True, help='固定的模型版本或快照')
    parser.add_argument('--reasoning-effort', default='high', choices=['none', 'low', 'medium', 'high', 'xhigh', 'max'])
    parser.add_argument('--cli-path', help='可执行文件路径；测试可指向假的 CLI')
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--failures', type=Path)
    parser.add_argument('--no-cache', action='store_true', help='绕过模型缓存，重新调用；已有产物仍遵循跳过/覆盖规则')
    return parser


def cli_json(args, prompt, version, validate, *, stage, template):
    executable = args.cli_path or args.annotator
    if args.cli_path:
        executable = str(Path(executable).resolve())
    cache = ModelCache(stage, dict(name=args.annotator, executable=executable, prompt_version=version), args.model,
                       dict(reasoning_effort=args.reasoning_effort, temperature=None,
                            enable_thinking=None, max_tokens=None), template, prompt,
                       enabled=not getattr(args, 'no_cache', False))
    cached = cache.load(validate)
    if cached is not None:
        return cached
    if args.annotator == 'codex':
        command = [executable, 'exec', '-m', args.model, '-c', 'model_reasoning_effort=' + args.reasoning_effort,
                   '-s', 'read-only', '--skip-git-repo-check', '--ephemeral', '--color', 'never', '-']
    else:
        command = [executable, '-p', '--model', args.model, '--effort', args.reasoning_effort,
                   '--tools', '', '--no-session-persistence']
    TMP.mkdir(exist_ok=True)
    # 盲标 CLI 不在数据目录运行；提示词仅经 stdin 传入，不给 runs 路径。
    # Claude 禁用工具。Codex 的 read-only 沙箱不是读取隔离，仍需可信的 CLI/模型遵守盲标协议。
    with tempfile.TemporaryDirectory(prefix='model-cli-', dir=TMP) as cwd:
        raws = []
        for attempt in range(2):
            actual_prompt = prompt + ('\n上次输出无效，请重新按指定 JSON 格式输出。' if attempt else '')
            result = subprocess.run(command, input=actual_prompt,
                                    text=True, capture_output=True, cwd=cwd, timeout=args.timeout)
            if result.returncode:
                # 记下 CLI 的退出码和最后几行输出（不含提示词），便于排查上下文超限、额度、鉴权等问题
                tail = ' | '.join(x for x in ((result.stdout or '') + '\n' + (result.stderr or '')).strip().splitlines() if x.strip())[:400]
                raise RuntimeError(f'CLI 退出失败（退出码 {result.returncode}）：{tail}')
            # 框架第 12 节第 7 层：模型原始返回一律留存在 tmp/model-raw/（不进仓库），身份记录里写路径和哈希
            raws.append(save_raw(result.stdout))
            raws[-1]['input_sha256'] = sha256_text(actual_prompt)
            try:
                parsed = parse_json(result.stdout)
                value = validate(copy.deepcopy(parsed))
            except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
                if attempt:
                    raise ModelOutputError(f'两次 CLI 输出均未通过 JSON/结构校验；最后一次：{exc}；原始返回：{raws[-1]["path"]}')
                continue
            # 框架第 12.3 节：CLI 版本、推理强度、提示词哈希、输入规模；CLI 不报告 token 和上下文上限，如实记“未记录”
            identity = dict(tool=args.annotator, cli_version=cli_version(executable), model=args.model,
                            reasoning_effort=args.reasoning_effort, prompt_version=version,
                            prompt_sha256=sha256_text(prompt), input_sha256=sha256_text(prompt),
                            template_sha256=sha256_text(template), input_chars=len(prompt),
                            context=dict(window='未记录', input_tokens='未记录', input_truncated='未记录'),
                            sandbox='read-only' if args.annotator == 'codex' else 'tools disabled',
                            attempts=attempt + 1, raw_outputs=raws, environment=environment(), time=now())
            cache.save(parsed, identity)
            return value, identity


def prompt_file(name):
    return Path(__file__).with_name('prompts').joinpath(name + '.md').read_text(encoding='utf-8')
