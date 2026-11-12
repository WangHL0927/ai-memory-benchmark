"""模型成功结果的内容寻址缓存；缓存及原始返回始终只落在 tmp/。"""
import copy
import json
from uuid import uuid4

from tools.config import sha256_file, sha256_text
from tools.paths import ROOT, TMP
from tools.trial import now, set_cache_status, write_json

CACHE_ROOT = TMP / 'cache'


def canonical_json(value):
    """固定键序、分隔符和编码；拒绝 NaN 等非 JSON 数值。"""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def save_raw(value):
    """逐次保存完整返回，随机后缀避免并发请求覆盖同一份原文。"""
    content = value.encode('utf-8') if isinstance(value, str) else value
    path = TMP / 'model-raw' / f"{now().replace(':', '')}-{uuid4().hex}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return dict(path=str(path.relative_to(ROOT)), sha256=sha256_file(path))


class ModelCache:
    """只接收通过业务校验的模型 JSON；命中时重新校验并重建当前样本外壳。"""

    def __init__(self, stage, tool, model, parameters, template, request, enabled=True):
        if stage not in ('annotate', 'judge'):
            raise ValueError('缓存阶段必须为 annotate 或 judge')
        self.inputs = dict(stage=stage, tool=tool, model=model, parameters=parameters,
                           template_sha256=sha256_text(template),
                           input_sha256=sha256_text(request if isinstance(request, str) else canonical_json(request)))
        self.key = sha256_text(canonical_json(self.inputs))
        self.path = CACHE_ROOT / stage / self.key[:2] / (self.key + '.json')
        self.enabled = enabled
        set_cache_status('miss')

    def load(self, validate):
        if not self.enabled:
            return None
        try:
            record = json.loads(self.path.read_text(encoding='utf-8'))
            if record['key'] != self.key or record['inputs'] != self.inputs:
                return None
            if record['result_sha256'] != sha256_text(canonical_json(record['result'])):
                return None
            if not isinstance(record['identity'], dict) or not record['raw_outputs']:
                return None
            # 原文丢失或损坏时重新调用，不能把无法审计的记录当作命中。
            for raw in record['raw_outputs']:
                path = (ROOT / raw['path']).resolve()
                if not path.is_relative_to((TMP / 'model-raw').resolve()):
                    return None
                if sha256_file(path) != raw['sha256']:
                    return None
            result = validate(copy.deepcopy(record['result']))
            identity = record['identity']
            identity.update(cache='hit', cache_key=self.key, raw_outputs=record['raw_outputs'])
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
            # 缓存不是输入数据的事实来源；损坏或不再通过当前校验的条目按未命中处理。
            return None
        set_cache_status('hit')
        return result, identity

    def save(self, result, identity):
        identity.update(cache='miss', cache_key=self.key)
        if self.enabled:
            write_json(self.path, dict(key=self.key, inputs=self.inputs, result=result,
                                      result_sha256=sha256_text(canonical_json(result)), identity=identity,
                                      raw_outputs=identity['raw_outputs'], written_at=now()))
