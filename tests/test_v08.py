from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.model.v08.reader import K, Native, answer
from tools.model.v08.report import observed_read_tokens
from tools.trial import write_json


class NativeReadingTest(unittest.TestCase):
    def test_parallel_calls_respect_budget_and_force_final_answer(self):
        selected=[dict(id=f'c{i}',type='function',function=dict(name='lookup',arguments='{}'))
                  for i in range(7)]
        batch=dict(choices=[dict(finish_reason='tool_calls',message=dict(tool_calls=selected,content=None))])
        final=dict(choices=[dict(finish_reason='stop',message=dict(content='Answer'))])
        observed=[]
        def chat(config,payload,path,timeout):
            observed.append(payload.get('tool_choice'))
            return (batch,final)[len(observed)-1]
        dispatched=[]
        def call(name,args):
            dispatched.append((name,args))
            return {'data':'example'}
        tool=dict(type='function',function=dict(name='lookup',parameters={'type':'object'}))
        with tempfile.TemporaryDirectory() as d,patch('tools.model.v08.reader.chat',side_effect=chat):
            receipt=answer({},dict(question='Synthetic question'),Path(d),len,native=Native([tool],call,'initial'))
        self.assertEqual(receipt['status'],'answered')
        self.assertEqual(observed,['auto','none'])
        self.assertEqual(len(dispatched),K)
        self.assertEqual([row['step'] for row in receipt['delivered']],list(range(K+1)))
        self.assertEqual(receipt['read_tokens'],sum(len(row['content']) for row in receipt['delivered']))

    def test_overflow_bills_only_memory_reaching_the_reader(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            write_json(path/'answer.json',dict(status='reader_overflow',
                delivered=[dict(step=0,tokens=7),dict(step=1,tokens=11),dict(step=2,tokens=200_000)],
                reads=[dict(step=1,called=True),dict(step=2,called=True)]))
            messages=[dict(role='system'),dict(role='user'),dict(role='tool')]
            write_json(path/'reader-01.json',dict(status=200,request=dict(messages=messages)))
            self.assertEqual(observed_read_tokens(path),18)
