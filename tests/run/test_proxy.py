"""共享代理的参数透传、原始审计和失败处理；只连合成 HTTP 服务。"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from tools.paths import TMP
from tools.run.proxy import ModelProxy


@contextmanager
def upstream(reply, status=200, content_type='application/json'):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.end_headers()
            self.wfile.write(reply.encode() if isinstance(reply, str) else json.dumps(reply).encode())

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    config = dict(base_url=f'http://127.0.0.1:{server.server_port}/v1', key='SYNTHETIC-SECRET',
                  model='synthetic-llm', embedding_model='synthetic-embedding', temperature=0.2,
                  max_tokens=100, enable_thinking=False, dimensions=2)
    try:
        yield config, seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def post(proxy, endpoint, body):
    return urlopen(Request(proxy.url + '/' + endpoint, data=json.dumps(body).encode()), timeout=5)


class SharedProxyTests(unittest.TestCase):
    def test_product_fields_are_forwarded_unchanged_and_audited(self):
        from tools.run.openviking import ModelProxy as OriginalImport
        self.assertIs(ModelProxy, OriginalImport)
        response = dict(choices=[dict(finish_reason='stop')], usage=dict(prompt_tokens=3, completion_tokens=2, total_tokens=5))
        with upstream(response) as (config, seen), tempfile.TemporaryDirectory(dir=TMP) as temporary:
            body = dict(model=config['model'], messages=[dict(role='user', content='合成用户原文')],
                        chat_template_kwargs=dict(enable_thinking=True), reasoning_effort='high',
                        reasoning=dict(enabled=True), max_completion_tokens=999,
                        max_tokens=999, temperature=1.0, enable_thinking=True)
            with ModelProxy(config, Path(temporary), 5) as proxy:
                with post(proxy, 'chat/completions', body) as reply:
                    self.assertEqual(reply.status, 200)
                self.assertEqual(proxy.summary()['llm']['tokens']['total_tokens'], 5)
                self.assertEqual(proxy.errors, [])
            sent = seen[0]
            self.assertEqual(sent, body)
            audit = json.loads((Path(temporary) / 'call-0001.json').read_text())
            self.assertEqual(audit['product_request'], body)
            self.assertEqual(audit['request'], sent)
            self.assertIn('response_sha256', audit)
            self.assertNotIn(config['key'], json.dumps(audit))

    def test_embedding_dimensions_format_and_large_batch_are_not_overridden(self):
        response = dict(error='server rejects requested dimensions')
        with upstream(response, 400) as (config, seen), tempfile.TemporaryDirectory(dir=TMP) as temporary:
            body = dict(model=config['embedding_model'], input=['合成'] * 21, dimensions=999, encoding_format='base64')
            with ModelProxy(config, Path(temporary), 5) as proxy:
                with self.assertRaises(HTTPError) as error:
                    post(proxy, 'embeddings', body)
                self.assertEqual(error.exception.code, 400)
                self.assertEqual(json.loads(error.exception.read()), response)
                error.exception.close()
                self.assertEqual(proxy.calls[0]['status'], 400)
                self.assertEqual(proxy.errors, [])
                self.assertEqual(proxy.parameters()['embedding']['dimensions'], [dict(sent=True, value=999, count=1)])
            self.assertEqual(seen, [body])
            audit = json.loads((Path(temporary) / 'call-0001.json').read_text())
            self.assertEqual(json.loads(audit['response']), response)

    def test_parameter_distribution_distinguishes_absent_null_and_values(self):
        response = dict(choices=[], usage=dict(total_tokens=1))
        with upstream(response) as (config, seen), tempfile.TemporaryDirectory(dir=TMP) as temporary:
            with ModelProxy(config, Path(temporary), 5) as proxy:
                for fields in ({}, dict(temperature=None), dict(temperature=.3, reasoning=dict(enabled=False)),
                               dict(temperature=.3, reasoning=dict(enabled=False))):
                    post(proxy, 'chat/completions', dict(model=config['model'], messages=[], **fields)).close()
                params = proxy.parameters()['llm']
                self.assertEqual(params['temperature'], [dict(sent=False, value='未发送', count=1),
                                 dict(sent=True, value=.3, count=2), dict(sent=True, value=None, count=1)])
                self.assertEqual(params['stream'], [dict(sent=False, value='未发送', count=4)])
                self.assertEqual(params['reasoning'], [dict(sent=False, value='未发送', count=2),
                                 dict(sent=True, value=dict(enabled=False), count=2)])

    def test_stream_options_and_non_json_errors_are_forwarded(self):
        sse = 'data: {"choices":[{"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        with upstream(sse, content_type='text/event-stream') as (config, seen), tempfile.TemporaryDirectory(dir=TMP) as temporary:
            body = dict(model=config['model'], messages=[], stream=True)
            with ModelProxy(config, Path(temporary), 5) as proxy:
                with post(proxy, 'chat/completions', body) as reply:
                    self.assertEqual(reply.read().decode(), sse)
                self.assertEqual(proxy.summary()['llm']['tokens']['total_tokens'], '未记录')
                self.assertEqual(proxy.calls[0]['finish_reasons'], ['stop'])
            self.assertEqual(seen, [body])  # 不能为了计量插入 stream_options。
        with upstream('synthetic upstream error', 400, 'text/plain') as (config, seen), tempfile.TemporaryDirectory(dir=TMP) as temporary:
            with ModelProxy(config, Path(temporary), 5) as proxy:
                with self.assertRaises(HTTPError) as error:
                    post(proxy, 'chat/completions', dict(model=config['model'], messages=[]))
                self.assertEqual(error.exception.code, 400)
                self.assertEqual(error.exception.headers['Content-Type'], 'text/plain')
                self.assertEqual(error.exception.read(), b'synthetic upstream error')
                error.exception.close()
                self.assertEqual(proxy.errors, [])

    def test_http_error_is_counted_and_key_redacted(self):
        with upstream(dict(error='回显 SYNTHETIC-SECRET'), status=500) as (config, _), tempfile.TemporaryDirectory(dir=TMP) as temporary:
            with ModelProxy(config, Path(temporary), 5) as proxy:
                with self.assertRaises(HTTPError) as error:
                    post(proxy, 'chat/completions', dict(model=config['model'], messages=[]))
                error.exception.close()
                self.assertEqual(proxy.calls[0]['status'], 500)
                self.assertEqual(proxy.summary()['llm']['calls'], 1)
            audit = (Path(temporary) / 'call-0001.json').read_text()
            self.assertNotIn(config['key'], audit)
            self.assertIn('[REDACTED]', audit)

    def test_wrong_model_never_reaches_upstream_and_is_audited(self):
        with upstream({}) as (config, seen), tempfile.TemporaryDirectory(dir=TMP) as temporary:
            with ModelProxy(config, Path(temporary), 5) as proxy:
                with self.assertRaises(HTTPError) as error:
                    post(proxy, 'chat/completions', dict(model='wrong', messages=[]))
                error.exception.close()
                self.assertEqual(len(proxy.errors), 1)
            self.assertEqual(seen, [])
            audit = json.loads(next(Path(temporary).glob('proxy-error-*.json')).read_text())
            self.assertEqual(audit['product_request']['model'], 'wrong')


if __name__ == '__main__':
    unittest.main()


class ThrottleTests(unittest.TestCase):
    def test_throttled_request_is_retried_with_backoff(self):
        replies = [(429, dict(error='Too many requests')),
                   (200, dict(choices=[dict(message=dict(content='ok'), finish_reason='stop')],
                              usage=dict(prompt_tokens=1, completion_tokens=1, total_tokens=2)))]
        hits = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                status, body = replies[min(len(hits), 1)]
                hits.append(status)
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True).start()
        config = dict(base_url=f'http://127.0.0.1:{server.server_port}/v1', key='SYNTHETIC-SECRET',
                      model='synthetic-llm', embedding_model='synthetic-embedding', temperature=1.0,
                      max_tokens=100, enable_thinking=False, dimensions=2)
        from unittest import mock
        try:
            with mock.patch('tools.run.proxy.time.sleep') as sleep, tempfile.TemporaryDirectory(dir=TMP) as temporary:
                with ModelProxy(config, Path(temporary), 5) as proxy:
                    post(proxy, 'chat/completions', dict(model=config['model'], messages=[])).close()
                    self.assertEqual(hits, [429, 200])
                    self.assertEqual(proxy.calls[0]['throttle_retries'], 1)
                sleep.assert_called_once()
        finally:
            server.shutdown()
            server.server_close()
