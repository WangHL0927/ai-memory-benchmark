"""从 OpenViking 的 run 派生“只看摘要”版本：去掉记忆正文里产品附带的原始聊天记录段。

OpenViking 会在记忆文件末尾附上 `# <日期> ChatLog:` 原文段。主表照常计分；本脚本生成的
`<system>-summaryonly` 只作补充对照，便于和做提炼的产品比较。不调用模型、不改原 run。
"""
import argparse
import re
from pathlib import Path

from tools.config import sha256_file
from tools.trial import output_path, read_json, write_json

SOURCE = 'openviking-0.5.4.1-ltm-defaults'
CHATLOG = re.compile(r'^#+ [^\n]*ChatLog', re.M)


def strip(statement):
    match = CHATLOG.search(statement)
    return (statement[:match.start()] if match else statement).strip()


def derive(run, source_sha256):
    system = dict(run['system'], name=run['system']['name'] + '-summaryonly',
                  derived=dict(source=run['system']['name'], source_sha256=source_sha256,
                               transform='删去记忆正文中从 “# … ChatLog” 标题起的原始聊天记录段；删后为空的记忆丢弃'))
    entries = []
    for entry in run['D']:
        text = strip(entry['draft']['statement'])
        if text:
            entries.append(dict(entry, draft=dict(entry['draft'], statement=text)))
    return dict(run, system=system, D=entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--system', default=SOURCE)
    args = parser.parse_args()
    count = 0
    for path in sorted(args.data_dir.glob(f'*/runs/{args.system}.json')):
        target = output_path(path.with_name(args.system + '-summaryonly.json'), args.data_dir)
        write_json(target, derive(read_json(path), sha256_file(path)))
        count += 1
    print('已生成', count)


if __name__ == '__main__':
    assert strip('# A\nx\n\n# 2026-11-11 (Wed) ChatLog:\n**user**: y') == '# A\nx'
    assert strip('no log') == 'no log'
    main()
