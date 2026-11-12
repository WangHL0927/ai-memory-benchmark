"""读取仓库根目录的 .env（只留本地、不提交）和运行环境信息（框架第 12 节）。"""
from functools import lru_cache
import hashlib
from pathlib import Path
import os
import platform
import subprocess
import sys

from tools.paths import ROOT


def load_env(path=None):
    """把 .env 里的 KEY=VALUE 放进环境变量；已存在的环境变量优先，不被覆盖。"""
    # EVAL_DOTENV 可指向别的文件；测试把它指向不存在的路径，避免读到本机真实配置
    path = path or Path(os.environ.get('EVAL_DOTENV') or ROOT / '.env')
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip())


def sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    """运行清单里的环境部分：仓库 commit、是否有未提交改动、Python、操作系统。"""
    def git(*args):
        try:
            return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    first = (ROOT / 'docs' / 'framework.md').read_text(encoding='utf-8').splitlines()[0]
    return dict(framework_doc=first.lstrip('# ').strip(), repo_commit=git('rev-parse', 'HEAD') or '未记录',
                repo_dirty=bool(git('status', '--porcelain')),
                python=sys.version.split()[0], platform=platform.platform())


@lru_cache(maxsize=None)
def cli_version(executable):
    try:
        out = subprocess.run([executable, '--version'], capture_output=True, text=True, timeout=30).stdout.strip()
        return out.splitlines()[0] if out else '未记录'
    except (OSError, subprocess.SubprocessError):
        return '未记录'


def _short_hash(x):
    """模板哈希前 8 位；拆分判定把召回、精确两份模板的哈希存成字典，拼在一起显示。"""
    t = x.get('template_sha256') or (x.get('prompt_sha256') if x.get('input_sha256') else None) or ''
    if isinstance(t, dict):
        return '/'.join(str(v)[:8] for v in t.values())
    return str(t)[:8]


def summarize_identities(items):
    """把标注者 / 判定者 / 被测对象的身份记录按“同一套配置”分组，返回一行一组的可读摘要；完整记录留在 JSON。"""
    if not items:
        return ['未记录']
    if isinstance(items, dict):  # 被测对象：{系统名: [身份, ...]}
        return [f"{name}：" + '；'.join(summarize_identities(group)) for name, group in items.items()]
    groups = {}
    for x in items:
        if not isinstance(x, dict):
            continue
        p = x.get('parameters') or {}
        key = (x.get('tool') or x.get('name'), x.get('llm') or x.get('model'), x.get('reasoning_effort'),
               p.get('temperature'), p.get('enable_thinking'), x.get('cli_version'), x.get('prompt_version'),
               _short_hash(x))
        groups[key] = groups.get(key, 0) + 1
    lines = []
    for (tool, model, effort, temp, thinking, cli, prompt, tpl), n in groups.items():
        parts = [f"{tool or '—'} {model or '无模型'}"]
        if effort: parts.append(f"推理强度 {effort}")
        if temp is not None: parts.append(f"温度 {temp}")
        if thinking is not None: parts.append('思考 ' + ('开' if thinking else '关'))
        if cli: parts.append(cli)
        if prompt: parts.append(f"提示词 {prompt}" + (f"（模板 {tpl}…）" if tpl else ''))
        parts.append(f"{n} 次")
        lines.append(' · '.join(parts))
    return lines or ['未记录']
