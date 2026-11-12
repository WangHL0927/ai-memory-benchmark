import json
import tempfile
import unittest
from pathlib import Path

from tools.trial import batch, set_cache_status
from tests.helpers import TMP


class BatchTests(unittest.TestCase):
    def test_circuit_breaker_and_progress_log(self):
        calls = []

        def worker(task):
            calls.append(task)
            raise ValueError('合成的系统性错误')

        with tempfile.TemporaryDirectory(dir=TMP) as d:
            failures = Path(d) / 'f.json'
            self.assertEqual(batch(range(10), worker, failures, max_consecutive_failures=3), 1)
            record = json.loads(failures.read_text())
            self.assertEqual(len(calls), 3)          # 连续失败 3 次后停止派发
            self.assertTrue(record['circuit_broken'])
            self.assertEqual(record['not_run'], 7)
            log = Path(record['log']).read_text()
            self.assertIn('原因=合成的系统性错误', log)
            self.assertIn('cache=miss', log)
            self.assertEqual(log.count('未执行（已熔断）'), 7)
            Path(record['log']).unlink()

    def test_concurrent_cache_status_does_not_leak_between_samples(self):
        from threading import Barrier
        barrier = Barrier(2)

        def worker(task):
            set_cache_status('hit' if task == 'cached' else 'miss')
            barrier.wait(timeout=5)
            return 'written'

        with tempfile.TemporaryDirectory(dir=TMP) as d:
            failures = Path(d) / 'f.json'
            self.assertEqual(batch(['cached', 'fresh'], worker, failures, concurrency=2), 0)
            log_path = Path(json.loads(failures.read_text())['log'])
            log = log_path.read_text()
            self.assertIn('完成：cached 状态=written cache=hit', log)
            self.assertIn('完成：fresh 状态=written cache=miss', log)
            log_path.unlink()


if __name__ == '__main__':
    unittest.main()
