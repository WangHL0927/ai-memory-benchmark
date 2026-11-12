"""拆分判定 report.json 的自包含 HTML。"""
import argparse
from html import escape
import json
from pathlib import Path

from tools.paths import TMP
from tools.score_split import LABELS, METRIC_KEYS, cell, main_tables, identity_summary, system_identity_lines
from tools.trial import output_path, read_json

CSS = '''
:root{--bg:#f6f7f9;--surface:#fff;--text:#202530;--muted:#687181;--line:#e1e5eb;--accent:#3769c8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.65 system-ui,sans-serif}
main{max-width:1180px;margin:48px auto;padding:0 24px}h1{font-size:30px;letter-spacing:-.5px;margin-bottom:8px}
h2{font-size:21px;margin:36px 0 14px}h3{font-size:16px}p,.muted{color:var(--muted)}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:22px;margin:18px 0}
dl{display:grid;grid-template-columns:130px 1fr;gap:8px;margin:0}dt{color:var(--muted)}dd{margin:0;overflow-wrap:anywhere}
.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:13px 14px;text-align:right;border-bottom:1px solid var(--line);min-width:170px}
th:first-child,td:first-child{text-align:left;min-width:220px}thead th{background:#f3f6fb;color:var(--accent)}
small{display:block;font-size:12px;color:var(--muted)}a{color:var(--accent)}pre{white-space:pre-wrap;overflow-wrap:anywhere}
.notice{border-left:3px solid var(--accent);padding-left:14px}footer{margin:40px 0;color:var(--muted)}
@media(max-width:650px){main{margin:24px auto;padding:0 12px}.panel{padding:14px}dl{grid-template-columns:1fr;gap:2px}dd{margin-bottom:10px}}
'''


def metadata(value):
    return escape(json.dumps(value, ensure_ascii=False, indent=2)) if value else '未记录'


def render(report):
    parts = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>拆分判定报告</title><style>' + CSS + '</style></head><body><main>',
             '<h1>记忆摄取评测 · 拆分判定</h1>',
             '<p>协议 split-v1 · 数据集：' + escape('、'.join(report['datasets'])) +
             f" · 样本数 {report['sample_count']} · 会话级 bootstrap {report['bootstrap']['iterations']} 次，种子 {report['bootstrap']['seed']}，95% 区间</p>",
             '<p class="notice">' + escape(report['calibration_notice']) + '</p>']
    parts += ['<section class="panel"><h2>被测对象与模型参数</h2><ul>']
    parts += ['<li>' + escape(line) + '</li>' for line in system_identity_lines(report)]
    parts += ['</ul><p>完整参数、提示词哈希和运行环境见 report.json。</p></section>']
    for title, key in (('标注者身份', 'annotators'), ('判定者身份', 'judges')):
        summary = identity_summary(report[key])
        parts += ['<section class="panel"><h2>' + title + '</h2><pre>' + metadata(summary) + '</pre></section>']
    if report.get('agreement'):
        parts += ['<section class="panel"><h2>人工一致率</h2><pre>' + metadata(report['agreement']) + '</pre></section>']
    parts += ['<p class="notice">' + escape(w) + '</p>' for w in report['warnings']]
    for dataset, table in report['datasets'].items():
        parts += ['<h2>数据集：' + escape(dataset) + f"（{table['sample_count']} 个样本）</h2>",
                  '<p>框架 / 政策：' + metadata(table['protocol_versions']) + '</p>']
        for title, rows in main_tables(table):
            keys = METRIC_KEYS
            parts += ['<section class="panel"><h3>' + escape(title) + ' / 主口径（排除 uncertain）' + '</h3>',
                      '<div class="scroll"><table><thead><tr><th>系统</th><th>有效样本 / 会话</th><th>排除样本</th>']
            parts += ['<th>' + LABELS[k] + '</th>' for k in keys]
            parts += ['</tr></thead><tbody>']
            for _, result, summary in rows:
                if summary is None:
                    continue
                parts += ['<tr><th scope="row">' + escape(result['display_name']) + '</th>',
                          f"<td>{summary['counts'].get('samples', 0):g} / {summary['clusters']}</td><td>{result['failed_count']}</td>"]
                parts += ['<td>' + escape(cell(summary['metrics'][k])) + '</td>' for k in keys]
                parts += ['</tr>']
            parts += ['</tbody></table></div></section>']
    parts += [f"<h2>失败或缺失：{report['failure_count']} 个系统 × 样本对</h2>", '<p>单独列出，不计入分母，也不当作 0 分。</p><ul>']
    parts += ['<li>' + escape(' / '.join(str(f[k]) for k in ('dataset', 'system', 'sample_id', 'code', 'reason'))) + '</li>'
              for f in report['failures']]
    parts += ['</ul><h2>产品与原生流程的差异</h2>',
              '<p>Basic Memory 标注为“Basic Memory + 固定 agent 的组合”，单独展示，不与产品并排排序。以下内容取自各 run 的 system 元信息；未记录项不推断。</p>']
    for name, differences in report['native_differences'].items():
        parts += ['<section class="panel"><h3>' + escape(name) + '</h3><pre>' + metadata(differences) + '</pre></section>']
    parts += ['<footer><p>比率均附分子 / 分母及 95% 区间；字符数按 Python Unicode 字符计数。',
              '召回按单元判断全部记忆的联合支持，精确误记率按陈述计，原文摘录整体计 1 个陈述；按条口径保留在 JSON，政策价值判断独立列出。',
              '随机与定向样本混合时分表。完整分层及样本粒度见 JSON。</p></footer></main></body></html>']
    return ''.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, default=TMP / 'trial/report-split.html')
    args = parser.parse_args()
    try:
        report = read_json(args.report)
        if report.get('protocol') != 'split-v1':
            raise ValueError('不是拆分判定报告')
        target = output_path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render(report), encoding='utf-8')
        print(f'已输出 {target}')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'错误：{exc}\n')


if __name__ == '__main__':
    main()
