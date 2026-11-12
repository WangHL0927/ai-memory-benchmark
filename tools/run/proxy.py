"""被测产品共用的模型代理：校验指定模型、计量并保存脱敏审计；不改写消息。"""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time
import os
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid

from tools.config import load_env, sha256_text
from tools.paths import ROOT
from tools.trial import now, write_json


LLM_FIELDS = ('temperature', 'max_tokens', 'max_completion_tokens', 'enable_thinking', 'reasoning',
              'reasoning_effort', 'chat_template_kwargs', 'response_format', 'stream', 'stream_options', 'top_p', 'seed')

def settings():
    load_env()
    names = ('SUT_BASE_URL', 'SUT_API_KEY', 'SUT_LLM_MODEL',
             'SUT_EMBEDDING_MODEL', 'SUT_EMBEDDING_DIMENSIONS')
    if any(not os.environ.get(n) for n in names):
        raise ValueError('缺少必要环境变量：' + ', '.join(n for n in names if not os.environ.get(n)))
    result = dict(base_url=os.environ['SUT_BASE_URL'].rstrip('/'),
                  key=os.environ['SUT_API_KEY'], model=os.environ['SUT_LLM_MODEL'],
                  embedding_model=os.environ['SUT_EMBEDDING_MODEL'],
                  dimensions=int(os.environ['SUT_EMBEDDING_DIMENSIONS']))
    if result['dimensions'] <= 0:
        raise ValueError('embedding 维度必须为正')
    return result


class ModelProxy:
    """只在内存持有真实 key；不保存 HTTP headers，不让产品进程接触 key。"""
    def __init__(self, config, directory, timeout):
        self.config, self.directory, self.timeout = config, directory, timeout
        self.calls, self.lock = [], threading.Lock()
        self.errors = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                record, original_body = {}, None
                try:
                    body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    original_body = json.loads(json.dumps(body))
                    kind = {'/v1/chat/completions': 'llm', '/v1/embeddings': 'embedding'}.get(self.path)
                    if kind is None:
                        raise RuntimeError('非预期模型路径')
                    expected = config['model'] if kind == 'llm' else config['embedding_model']
                    if body.get('model') != expected:
                        raise RuntimeError('产品请求了非指定模型')
                    with owner.lock:
                        index = len(owner.calls) + 1
                        record.update(kind=kind, index=index, time=now(), status='pending',
                                      parameters={k: body[k] for k in (LLM_FIELDS if kind == 'llm' else ('dimensions',)) if k in body})
                        owner.calls.append(record)
                    payload = json.dumps(body, ensure_ascii=False).encode()
                    request = Request(config['base_url'] + self.path.removeprefix('/v1'), data=payload,
                                      headers={'Content-Type': 'application/json',
                                               'Authorization': 'Bearer ' + config['key']})
                    # 服务端限流（429，或 503 / ServiceUnavailable）时在代理里退避重试，最多 3 次，产品侧无感；
                    # 重试次数写进记录。其他错误原样返回给产品。
                    for retry in range(4):
                        try:
                            with urlopen(request, timeout=owner.timeout) as response:
                                raw, status = response.read(), response.status
                                content_type = response.headers.get('Content-Type', 'application/json')
                        except HTTPError as exc:
                            with exc:
                                raw, status = exc.read(), exc.code
                                content_type = exc.headers.get('Content-Type', 'application/json')
                        throttled = status == 429 or (status == 503 and b'ServiceUnavailable' in raw)
                        if not throttled or retry == 3:
                            break
                        time.sleep(5 * 2 ** retry)
                    record['throttle_retries'] = retry
                    # 原始请求/响应只在 tmp，万一上游错误回显 key 也先替换。
                    safe = raw.decode('utf-8', errors='replace').replace(config['key'], '[REDACTED]')
                    audit = owner.directory / f'call-{index:04d}.json'
                    record.update(status=status, request_sha256=hashlib.sha256(payload).hexdigest(),
                                  response_sha256=sha256_text(safe))
                    # 即使返回畸形 JSON 或错误维度，也先留存线上原始返回。
                    owner.audit(audit, dict(product_request=original_body, request=body, response=safe, **record))
                    record['raw_path'] = str(audit.relative_to(ROOT))
                    values = []
                    try:
                        if 'text/event-stream' in content_type:
                            values = [json.loads(line[6:]) for line in safe.splitlines()
                                      if line.startswith('data: ') and line[6:] != '[DONE]']
                        else:
                            values = [json.loads(safe)]
                        values = [v for v in values if isinstance(v, dict)]
                    except ValueError:
                        pass  # 计量缺失不改变服务端响应。
                    usage = next((v['usage'] for v in reversed(values) if v.get('usage')), None)
                    finishes = [c.get('finish_reason') for v in values for c in (v.get('choices') or [])
                                if isinstance(c, dict) and c.get('finish_reason')]
                    record.update(status=status, usage=usage or '未记录', finish_reasons=finishes,
                                  returned_models=sorted({v['model'] for v in values if v.get('model')}),
                                  request_sha256=hashlib.sha256(payload).hexdigest(),
                                  response_sha256=sha256_text(safe))
                    owner.audit(audit, dict(product_request=original_body, request=body, response=safe, **record))
                    record['raw_path'] = str(audit.relative_to(ROOT))
                    self.send_response(status)
                    self.send_header('Content-Type', content_type)
                    self.send_header('Content-Length', str(len(safe.encode('utf-8'))))
                    self.end_headers()
                    self.wfile.write(safe.encode('utf-8'))
                except Exception as exc:
                    record.update(status='proxy_error', error_type=type(exc).__name__)
                    if isinstance(exc, RuntimeError):
                        record['detail'] = str(exc)
                    with owner.lock:
                        owner.errors.append(record)
                    owner.audit(owner.directory / ('proxy-error-' + uuid.uuid4().hex + '.json'),
                               dict(product_request=original_body, **record))
                    self.send_response(502)
                    self.end_headers()
                    self.wfile.write(b'{"error":"adapter proxy failed"}')

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .05}, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    @property
    def url(self):
        return f'http://127.0.0.1:{self.server.server_port}/v1'

    def summary(self):
        result = {}
        for kind in ('llm', 'embedding'):
            rows = [c for c in self.calls if c['kind'] == kind]
            keys = ('prompt_tokens', 'completion_tokens', 'total_tokens')
            result[kind] = dict(calls=len(rows), tokens={
                k: sum(c['usage'].get(k, 0) for c in rows) if rows and all(isinstance(c.get('usage'), dict) and k in c['usage'] for c in rows) else '未记录'
                for k in keys})
        return result


    def audit(self, path, value):
        safe = json.dumps(value, ensure_ascii=False).replace(self.config['key'], '[REDACTED]')
        write_json(path, json.loads(safe))

    def parameters(self):
        """按审计记录汇总实际发送字段；缺字段和显式 null 分开计数。"""
        result = {}
        for kind, fields in (('llm', LLM_FIELDS), ('embedding', ('dimensions',))):
            rows = [c['parameters'] for c in self.calls if c['kind'] == kind]
            result[kind] = {}
            for field in fields:
                counts = {}
                for row in rows:
                    sent = field in row
                    value = row[field] if sent else '未发送'
                    key = json.dumps([sent, value], sort_keys=True, ensure_ascii=False)
                    counts[key] = counts.get(key, 0) + 1
                result[kind][field] = [dict(sent=json.loads(k)[0], value=json.loads(k)[1], count=n)
                                       for k, n in sorted(counts.items())]
        return result
