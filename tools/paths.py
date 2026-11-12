"""仓库路径与写入规则：不适合公开的内容只能写进被 git 忽略的 tmp/。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "tmp"
PRIVATE_DATA = TMP / "private" / "data"
PUBLIC_DATA = ROOT / "data"


def inside_repo(path):
    path = Path(path).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError("输出路径必须在本仓库内")
    return path


def private_path(path):
    """自有会话及其派生物只能写进 tmp/，或者仓库以外的地方。"""
    path = Path(path).resolve()
    tmp = TMP.resolve()
    if (path == ROOT or ROOT in path.parents) and path != tmp and tmp not in path.parents:
        raise ValueError("不适合公开的数据不能写进仓库的受跟踪目录，请放在 tmp/ 下（默认 tmp/private/data）")
    return path


def is_public(path):
    path, public = Path(path).resolve(), PUBLIC_DATA.resolve()
    return path == public or public in path.parents
