"""Render v0.8 from frozen private receipts without publishing source material."""
import argparse
from collections import Counter
from datetime import date
from functools import cache
from itertools import product as combinations
from pathlib import Path
from statistics import mean

from tools.model.v07.cli import JUDGE_MODEL, READER_MODEL, READER_WINDOW
from tools.model.v07.scoring import (
    ITERATIONS, READ_COEFFICIENTS, WRITE_COEFFICIENTS, log_penalty, resampling, value_and_cost,
)
from tools.model.v08.corpus import ordered
from tools.model.v08.products import PRODUCTS
from tools.paths import private_path
from tools.trial import read_json


def load(path, default=None):
    return read_json(path) if path.is_file() else default


def as_text(value):
    return 'NA' if value is None else f'{value:.2f}'


def fraction(n,d):
    return f'{n}/{d} ({n/d:.1%})' if d else 'NA (0/0)'


def domain(card):
    return 'coding' if card['dataset']=='trace-commons' else card['group']


def source_dir(root,product,card,mode):
    if product=='baseline':
        return root/'baselines'/card['id']/mode
    return root/product/domain(card)/'tasks'/card['id']/mode


def paired_credit(root,cards,left,right):
    differences=[]
    for card in cards:
        judgments=[load(source_dir(root,p,card,mode)/'judgment.json',{}).get('judgment')
                   for p,mode in (left,right)]
        if all(judgments):
            credits=[mean(row['status']=='supported' for row in judgment['requirements'])
                     for judgment in judgments]
            differences.append(credits[0]-credits[1])
    return len(differences),as_text(100*mean(differences) if differences else None)


@cache
def observed_read_tokens(path):
    receipt=load(path/'answer.json')
    if not receipt or not receipt['delivered']:
        return 0
    if receipt['status']=='answered':
        sent_steps={item['step'] for item in receipt['delivered']}
    else:
        completed=[row['request']['messages'] for f in sorted(path.glob('reader-*.json'))
                   if (row:=read_json(f))['status']==200]
        sent_steps={0,*range(1,1+max(sum(m['role']=='tool' for m in messages)
                                     for messages in completed))} if completed else set()
    product_steps={0,*(item['step'] for item in receipt['reads'] if item['called'])}
    return sum(item['tokens'] for item in receipt['delivered']
               if item['step'] in sent_steps & product_steps)


def row_for(card,verdict,read_tokens):
    stats=Counter()
    if verdict:
        scenario=card['scenario']
        points=verdict['requirements']
        hits=sum(p['status']=='supported' for p in points)
        stats.update({scenario+'_valid':1,scenario+'_credit':hits/len(points),
                      scenario+'_correct':verdict['verdict']=='correct',
                      verdict['verdict']:1,'accepted_requirements':hits,'requirements':len(points)})
        for flag in ('action_error','error_matches_memory','misleading','unclear_attribution','high_risk',
                     'incorrect_subject','superseded_used'):
            stats[flag]+=verdict.get(flag,False)
    return dict(outcomes=stats,num_tasks=1,valid_tasks=int(bool(verdict)),
                read_tokens=read_tokens,transcript_tokens=card['tokens'],write_multiple=0)


def writing(root,product,selection,scope):
    if product=='baseline':
        return dict(pw=0,attempted=0,written=0,failed=0,input=0,output=0,missing=0,calls=0,samples=[],
                    embedding_calls=0,embedding_input=0,embedding_unknown=0)
    subs=[i for i in selection['items'] if i['dataset']==scope]
    rows=[]
    for item in subs:
        path=root/product/domain(item)/'audits'/f"{item['id']}-ingest.json"
        if not path.is_file():
            continue
        capture=read_json(path)
        usage=capture['usage']['llm']
        embedding=capture['usage']['embedding']
        rows.append(dict(id=item['id'],input=usage['tokens']['prompt_tokens'],
                         output=usage['tokens']['completion_tokens'],status=capture['status'],
                         calls=usage['calls'],T=item['tokens'],embedding_calls=embedding['calls'],
                         embedding_input=embedding['tokens']['prompt_tokens']))
    observed=[i['input']/i['T'] for i in rows if i['input'] is not None]
    return dict(pw=mean(log_penalty(x,2.5) for x in observed) if len(observed)==len(subs) else None,
        attempted=len(rows),written=sum(r['status']=='written' for r in rows),
        failed=sum(r['status']!='written' for r in rows),input=sum(r['input'] or 0 for r in rows),
        output=sum(r['output'] or 0 for r in rows),missing=len(subs)-len(observed),
        calls=sum(r['calls'] for r in rows),samples=rows,embedding_calls=sum(r['embedding_calls'] for r in rows),
        embedding_input=sum(r['embedding_input'] or 0 for r in rows),
        embedding_unknown=sum(r['embedding_input'] is None for r in rows))


def coding_score(rows, write_penalty):
    score=value_and_cost(rows)
    score['write_penalty']=write_penalty
    v,r=score['value'],score['read_penalty']
    score['composite']=None if None in (v,r,write_penalty) else max(0,v-r-write_penalty)
    return score


def coding_interval(root, product, mode, cards, selection):
    """Resample the projects with their questions and ingestion costs kept together."""
    receipts={row['id']:row['input'] for row in writing(root,product,selection,'trace-commons')['samples']}
    projects=[]
    for group in dict.fromkeys(card['group'] for card in cards):
        rows=[]
        for card in cards:
            if card['group']!=group:
                continue
            qdir=source_dir(root,product,card,mode)
            record=load(qdir/'judgment.json',{})
            read_tokens=observed_read_tokens(qdir) if (qdir/'answer.json').is_file() else None
            rows.append(row_for(card,record.get('judgment'),read_tokens))
        penalties=[]
        for item in selection['items']:
            if item['group']==group:
                observed=0 if product=='baseline' else receipts.get(item['id'])
                penalties.append(log_penalty(observed/item['tokens'],2.5) if observed is not None else None)
        projects.append((rows,penalties))

    def composite(sampled):
        rows=[row for group,_ in sampled for row in group]
        penalties=[penalty for _,group in sampled for penalty in group]
        write_penalty=mean(penalties) if penalties and None not in penalties else None
        return {'composite':coding_score(rows,write_penalty)['composite']}

    return resampling(projects,composite)['composite']


def interval_text(interval):
    low,high=interval['ci95']
    if low is None or high is None:
        return 'NA'
    text=f'{low:.2f}–{high:.2f}'
    if interval['valid_resamples']<ITERATIONS:
        text+=f"（{interval['valid_resamples']:,}/{ITERATIONS:,} 次可计）"
    return text


def summarize(root,product,mode,cards,selection):
    statuses,delivery,ledger,diag,rows=Counter(),Counter(),Counter(),Counter(),[]
    scope_diag={}
    for card in cards:
        qdir=source_dir(root,product,card,mode)
        receipt=load(qdir/'answer.json',{})
        record=load(qdir/'judgment.json',{})
        verdict=record.get('judgment')
        status=receipt.get('status','untested')
        if status=='untested' and mode=='E' and (root/product/domain(card)/
                                                f"{card['group']}-export-error.json").is_file():
            status='export_unavailable'
        if status=='answered' and not verdict:
            status='judge_missing'
        statuses[status]+=1
        read_tokens=observed_read_tokens(qdir) if receipt else None
        if card['dataset']=='trace-commons':
            rows.append(row_for(card,verdict,read_tokens))
        if verdict:
            for key in ('incorrect_subject','superseded_used','action_error','misleading','unclear_attribution'):
                diag[key]+=verdict[key]
            diag['correct']+=verdict['verdict']=='correct'
            diag['memory_match_unused']+=verdict['error_matches_memory'] and not verdict['actual_reuse']
            diag['reader_action_error']+=verdict['action_error'] and not (
                verdict['misleading'] or verdict['unclear_attribution'] or verdict['error_matches_memory'])
            if card['abstention']:
                diag['abstention_answered']+=1
                diag['correct_abstention']+=verdict['verdict']=='correct'
            row_stats=[p['status']=='supported' for p in verdict['requirements']]
            diag['points']+=len(row_stats)
            diag['points_used']+=sum(row_stats)
        if product!='baseline':
            group=card['group']
            if group not in scope_diag:
                scope_diag[group]=load(root/product/domain(card)/f'{group}-diagnosis.json',{})
            stored=scope_diag[group].get('store_points',{})
            for p in card['requirements']:
                key=f"{card['id']}.{p['req_id']}"
                delivery['points']+=1
                delivery['store_known']+=key in stored
                delivery['present']+=stored.get(key,False)
            if verdict:
                for p in verdict['requirements']:
                    delivered=p['first_delivery_step']
                    key=f"{card['id']}.{p['requirement_id']}"
                    delivery['seen_known']+=1
                    delivery['step0']+=delivered==0
                    delivery['step1']+=delivered is not None and delivered<=1
                    delivery['step5']+=delivered is not None
                    present=stored.get(key,False)
                    delivery['present_in_judged']+=present
                    delivery['present_and_seen']+=present and delivered is not None
                    delivery['end_to_end']+=present and p.get('used_delivered_evidence',False)
        if receipt:
            ledger['read_receipts']+=1
            ledger['wall_s']+=receipt.get('elapsed_seconds',0)
            ledger['R']+=read_tokens
            for entry in receipt['reads']:
                ledger['tool_attempts']+=1
                ledger['tool_dispatched']+=entry['called']
                ledger['tool_errors']+=entry['is_error']
                ledger['tool_returns_R']+=entry['tokens']
            ledger['tool_capped']+=len(receipt['reads'])==5
            for call in receipt['calls']:
                ledger['reader_calls']+=1
                usage=call.get('usage') or {}
                ledger['reader_in']+=usage.get('prompt_tokens',0)
                ledger['reader_out']+=usage.get('completion_tokens',0)
                ledger['reader_usage_unknown']+=not usage
                ledger['reader_cached']+=(usage.get('prompt_tokens_details') or {}).get('cached_tokens',0)
            for call in load(qdir/'read-product-usage.json',[]):
                kind=call['kind']
                ledger['product_'+kind+'_calls']+=1
                usage=call.get('usage')
                if isinstance(usage,dict):
                    ledger['product_'+kind+'_tokens']+=usage.get('prompt_tokens',0)
                else:
                    ledger['product_'+kind+'_usage_unknown']+=1
    write=writing(root,product,selection,'trace-commons')
    scoring=coding_score(rows,write['pw'])
    attempted=sum(statuses[s] for s in ('answered','reader_overflow','reader_incomplete','product_read_failure'))
    return dict(status=statuses,diagnostic=diag,delivery=delivery,ledger=ledger,score=scoring,
                planned=len(cards),valid=statuses['answered'],attempted=attempted,correct=diag['correct'])


def table(lines, columns, rows):
    lines.extend(['','| '+' | '.join(columns)+' |','|'+'---|'*len(columns)])
    lines.extend('| '+' | '.join(map(str,row))+' |' for row in rows)


def render(root):
    selection,groups=ordered(root)
    formal=root/'formal'
    sets={g:read_json(root/'cards'/f'{g}.json') for g in groups}
    code=[q for g in ('tc11','tc12') for q in sets[g]['cards']]
    daily=[q for g in ('daily-1','daily-2') for q in sets[g]['cards']]
    assert len(code)==24 and len(daily)==12 and sum(q['export_subset'] for q in code+daily)==12
    systems=[*( (p,'N',p) for p in PRODUCTS),('baseline','no-memory','无记忆'),
             ('baseline','source-session','源会话对照')]
    rows={(p,mode):summarize(formal,p,mode,code,selection) for p,mode,label in systems}
    lines=[f'# 记忆累计库与原生读取评测 v0.8 · 基线与局限性报告 · {date.today().isoformat()}','',
           '## 这批数据能说明什么','',
           '在未用于调参的 28 段、36 题上，累计库与原生读取显示了五套产品配置在本轮的大致表现、'
           '证据从「库中有」到「送达」再到「使用」的分布，以及实际读写成本。','',
           '## 这批数据的局限','',
           '产品排名超出这批数据能支持的结论。','',
           '1. OpenViking 的 coding 仅 10/24 题可判，14 题超窗，其综合分对应可判子集，与其余产品的全题得分缺少可比性。',
           '2. X2 和 Basic Memory + agent 这 2 套配置每场只在会话结束触发 1 次摄取，'
           '用整场内容集中整理，相对原生轮次停点触发的偏差方向未知。',
           '3. 每套产品配置只有 1 次正式运行，每个 coding 场景计划 6 题，'
           '判词未作人工校准，综合分的 95% 项目聚类 bootstrap 区间见下表。',
           '4. X2 使用 v0.7.3，审核读取全量转录，写入成本和摄取提示词的后续改进未进入本轮。',
           '5. 摄取失败留在端到端结果中，HaluMem 的失败段为 X2 6/20、Basic Memory 4/20、ai-memory 2/20。',
           '6. 固定读者 qwen4-max 每题最多调用原生工具 5 次，X2 在 36/36 题触顶，结果体现该读者的读取方式。',
           '7. 中途修正了 Basic Memory 的 1 个 coding 域项目绑定并补跑，格式失败的判词沿原规则重判，'
           '部分产品的读侧嵌入用量不可观测。','',
           f'X2 是未公开的内部系统。主测 N 衡量固定 {READER_MODEL} 驱动各产品原生读路径的端到端表现。'
           'coding 为两个独立项目，各有四次连续写入。HaluMem 为两个独立用户，各十次连续会话。'
           '每题在期末正常可读状态启动独立读者会话。五个产品共用 24 道 coding 与 12 道 HaluMem 题。'
           'E 只覆盖事先选定的 12 道题。','',
           f"统一按 {selection['tokenizer']['encoding']}（tiktoken {selection['tokenizer']['version']}）计量 T/R，"
           f'设计作答窗口为 200,000 token，实际按更小的读者端点上限 {READER_WINDOW:,} token 执行。'
           '读者使用非流式 Chat Completions，思考开关使用端点默认。'
           f'题卡与金标用 {JUDGE_MODEL} high，判定用 {JUDGE_MODEL} medium。摄取用 deepseek-v4.1-flash-1017，'
           'X2 修订用 deepseek-v4-pro。','',
           '## Coding 主测 N']
    out=[]
    for p,mode,label in systems:
        b=rows[p,mode]
        s=b['score']
        interval=coding_interval(formal,p,mode,code,selection)
        out.append([label,f"{b['valid']}/{b['planned']}",as_text(s['value']),as_text(s['composite']),
                    interval_text(interval),
                    *((f"{s['scenarios'][k]['value']:.1%} ({s['scenarios'][k]['denominator']} 题)"
                       if s['scenarios'][k]['value'] is not None else 'NA (0 题)')
                      for k in ('S1','S2','S3','S4')),
                    fraction(b['diagnostic']['misleading'],b['valid']),
                    fraction(b['correct'],b['attempted']),fraction(b['correct'],b['planned']),
                    as_text(s['read_penalty']),as_text(s['write_penalty'])])
    table(lines,['配置','判定覆盖','价值分','综合分','综合分 95% 区间','S1','S2','S3','S4','S5 归因误导',
                 '可归因尝试全对','全题分母全对','读扣','写扣'],out)
    lines.extend(['','S1–S4 各计划 6 题，覆盖两个独立项目。取逐题要点完成比例的场景均值，每场景 25 分。'
                  'S5 仅扣可归因于实际交付记忆的行动错误。'
                  '读扣按每题依赖源段的去重 T，写扣对每段摄取的供应商 W/T 各计算一次再求均值。'
                  '主表保留判定覆盖、可归因尝试分母和全题分母。实验失败与未测保留为覆盖缺失。','',
                  f'区间对 2 个独立 coding 项目整组重采样 {ITERATIONS:,} 次，每次重算要点、误导扣分和读写成本，'
                  '只保留四场景均可计分的重采样。区间描述本轮项目构成的变化，重复运行和判定误差尚未测量。','',
                  '## Coding 综合分敏感性'])
    sensitivities=list(combinations(READ_COEFFICIENTS,WRITE_COEFFICIENTS))
    table(lines,['配置',*(f'{a}/{w}' for a,w in sensitivities)],[
         [label,*(as_text(max(0,s['value']-s['read_penalty']*a/4-s['write_penalty']*w/2.5)
                         if None not in (s['value'],s['read_penalty'],s['write_penalty']) else None)
                   for a,w in sensitivities)]
         for p,mode,label in systems if (s:=rows[p,mode]['score'])])
    lines.extend(['','列名为读扣系数 / 写扣系数。各组合使用同一份判词和实际 token。','',
                  '## 同权限导出作答 E'])
    out=[]
    for p in PRODUCTS:
        for label,data in (('coding',code),('HaluMem',daily)):
            subset=[q for q in data if q['export_subset']]
            z=summarize(formal,p,'E',subset,selection)
            paired=paired_credit(formal,subset,(p,'E'),(p,'N'))
            out.append([p,label,f"{z['valid']}/{z['planned']}",fraction(z['correct'],z['valid']),
                        fraction(z['diagnostic']['points_used'],z['diagnostic']['points']),
                        z['status']['reader_overflow'],z['status']['export_unavailable'],*paired,
                        z['ledger']['R'],z['ledger']['reader_in']])
    table(lines,['系统','数据域','有效 / 预选','已判定全对','要点完成','溢窗','不可导出','共同有效题',
                 'E−N 要点评分差（百分点）','R 合计','读者输入 token'],out)
    lines.extend(['','E 使用同一身份和项目的期末可见导出。超窗或无法完整导出属于 E 的覆盖缺失。'
                  'E 与 N 的差值包含输入形式、工具调用及读者处理长上下文带来的差异。','',
                  '## 原生集成与可见域'])
    table(lines,['配置','启动注入','原生读及 E 的范围'],[
        ['X2','本版本没有读取启动注入','同租户 agent、全局及授权团队，当前项目用于检索排序'],
        ['ai-memory 3.4.7','固定版本 SessionStart hook','当期项目和合法全局的 Wiki、会话观察和交接'],
        ['Basic Memory 1.0.0','本轮集成没有启动注入','原生项目笔记'],
        ['claude-mem 15.17.0','官方 worker 上下文','绑定项目的观察和会话摘要'],
        ['OpenViking 0.5.4.1','本轮集成没有启动注入','项目/用户命名空间的记忆及原生会话归档'],
    ])
    lines.extend(['','不同产品允许的共享范围由其原生入口决定。HaluMem 两个用户使用各自隔离的身份和存储。'
                  '期末记录数量按可见域统计，同一个合法共享条目可能出现在两个项目的诊断中。','',
                  '## HaluMem 分用户原生读与对照'])
    out=[]
    for p,mode,label in systems:
        for group in ('daily-1','daily-2'):
            qs=[q for q in daily if q['group']==group]
            b=summarize(formal,p,mode,qs,selection)
            per=[summarize(formal,p,mode,[q for q in qs if q['scenario']==s],selection)
                 for s in ('early','update','join')]
            out.append([label,group,f"{b['valid']}/6",
                       *(f"{z['correct']}/{z['valid']}（计划 {z['planned']}）" for z in per),
                       b['diagnostic']['incorrect_subject'],b['diagnostic']['misleading']])
    table(lines,['配置','用户','有效','早期事实全对','更新全对','跨段全对','错误主体','归因误导'],out)
    lines.extend(['','## 可见证据、使用和失败'])
    out=[]
    for p in PRODUCTS:
        for label,data in (('coding',code),('HaluMem',daily)):
            b=summarize(formal,p,'N',data,selection)
            d=b['delivery']
            s=b['status']
            out.append([p,label,f"{b['valid']}/{b['planned']}",fraction(d['present'],d['store_known']),
                *(fraction(d[f'step{k}'],d['seen_known']) for k in (0,1,5)),
                fraction(d['present_and_seen'],d['present_in_judged']),
                fraction(d['end_to_end'],d['points']),s['native_not_tested']+s['untested'],
                s['experiment_failure']+s['judge_missing'],s['reader_overflow'],
                s['product_read_failure'],s['reader_incomplete']])
    table(lines,['配置','域','有效','持久库中有','第 0 步','第 1 步累计','≤5 步','有存储且送达 / 有存储',
                 '端到端覆盖','未测','实验失败或缺判','溢窗','产品读失败','未生成终答'],out)
    out=[]
    for p in PRODUCTS:
        b=summarize(formal,p,'N',code,selection)
        q=b['diagnostic']
        out.append([p,fraction(q['correct_abstention'],q['abstention_answered']),
                    q['points_used'],q['action_error'],q['misleading'],q['unclear_attribution'],
                    q['memory_match_unused'],q['reader_action_error'],q['incorrect_subject'],q['superseded_used']])
    table(lines,['配置','正确暂缓','正确使用要点','行动错误','归因误导','归因不清','相符错误记忆未使用',
                 '无明确记忆来源的读者行动错误','错误主体','使用过期证据'],out)
    lines.extend(['','交付指标按实际可见文本的最早充分证据位置判定。仅返回 ID 保留为候选定位，内容展开后才可能算送达。'
                  '固有空库照常作答，正式评测前的试跑与正式用量分别计。','',
                  '## 读侧与写侧用量'])
    out=[]
    for p,mode,label in systems:
        b=summarize(formal,p,mode,code+daily,selection)
        l=b['ledger']
        out.append([label,l['reader_calls'],f"{l['reader_in']:,} / {l['reader_out']:,}",l['reader_cached'],
                    as_text(l['wall_s']/l['read_receipts'] if l['read_receipts'] else None),
                    as_text(l['tool_dispatched']/l['read_receipts'] if l['read_receipts'] else None),
                    fraction(l['tool_capped'],l['read_receipts']),l['tool_errors'],l['R'],
                    as_text(l['tool_returns_R']/l['tool_attempts'] if l['tool_attempts'] else None),
                    f"{l['product_llm_calls']} / {l['product_embedding_calls']}",
                    f"{l['product_llm_tokens']:,} / {l['product_embedding_tokens']:,}"])
    table(lines,['配置','读者生成调用','实际输入 / 输出 token','缓存输入 token','平均耗时（秒）','原生调用均值',
                 '触顶','工具错误','R 合计','每个工具回包 token 均值','可观测产品读 LLM / 嵌入调用',
                 '可观测产品读 LLM / 嵌入输入 token'],out)
    out=[]
    for p in PRODUCTS:
        for slug,label in (('trace-commons','coding'),('halumem','HaluMem')):
            w=writing(formal,p,selection,slug)
            out.append([p,label,w['attempted'],w['written'],w['failed'],w['calls'],
                        f"{w['input']:,} / {w['output']:,}",w['missing'],w['embedding_calls'],w['embedding_input']])
    table(lines,['配置','域','已尝试段','成功段','摄取失败段','写模型调用','W 输入 / 输出 token','输入未计量段',
                 '可观测写侧嵌入调用','可观测写侧嵌入输入 token'],out)
    failures=[]
    for p in PRODUCTS:
        for slug,label in (('trace-commons','coding'),('halumem','HaluMem')):
            w=writing(formal,p,selection,slug)
            failed=[r['id'] for r in w['samples'] if r['status']!='written']
            if failed:
                failures.append([p,label,'、'.join(failed)])
    if failures:
        table(lines,['配置','域','失败段标识'],failures)
    lines.extend(['','R 只累加每题新交付的原生注入和各次回包。窗口预检未发送的回包保留在本地回执。'
                  '实际读者输入累计包含每轮重发历史及工具定义。'
                  '本地嵌入未经过供应商代理，产品读用量列只覆盖可观测请求。','',
                  '## 期末摄取诊断'])
    out=[]
    for product in PRODUCTS:
        for g in groups:
            path=formal/product/('coding' if g in ('tc11','tc12') else g)
            exported=load(path/f'{g}-export.json')
            visible=len(exported) if exported is not None else 'NA'
            d=load(path/f'{g}-diagnosis.json')
            if not d:
                out.append([product,g,visible,'NA','NA','NA','NA','NA','NA'])
                continue
            units=d['store_units']
            counts=Counter()
            for batch in d['precision']:
                for row in batch['rows']:
                    for k in ('claims','unsupported','distorted','outdated_as_current','incorrect_scope'):
                        counts[k]+=row[k]
            out.append([product,g,visible,fraction(sum(units.values()),len(units)),counts['claims'],
                        counts['unsupported'],counts['distorted'],counts['outdated_as_current'],
                        counts['incorrect_scope']])
    table(lines,['配置','来源','期末可见条目','库内有证据金标','主张','无据','错误','旧值当现值','范围错误'],out)
    terminal=load(formal/'X2/terminal-audit.json',{})
    if terminal:
        out=[]
        for slug,prefix in (('coding','tc'),('HaluMem','daily-')):
            states,reasons=Counter(),Counter()
            drafted=0
            for name,info in terminal.items():
                if name.startswith(prefix):
                    drafted+=info['drafts']
                    states.update(info['states'])
                    reasons.update(info['reasons'])
            out.append([slug,drafted,states['approved'],states['rejected'],
                        sum(states.values())-states['approved']-states['rejected'],reasons['ai_failed']])
        table(lines,['X2 试验域','提交草案','审核通过','驳回','其他状态','AI 失败驳回'],out)
    lines.extend(['','## 共同题与局限'])
    paired=[]
    for i,p in enumerate(PRODUCTS):
        for other in PRODUCTS[i+1:]:
            paired.append([p,other,*paired_credit(formal,code,(p,'N'),(other,'N'))])
    table(lines,['配置 A','配置 B','共同有效 coding 题','要点评分均值差 A−B（百分点）'],paired)
    lines.extend(['','只有两个独立 coding 项目及两个 HaluMem 用户。题目在项目和用户内相关，配对差为同题描述。'
                  '分段来自两个真实父会话，同项目四段形成累计历史。HaluMem 两名用户曾有其他会话的历史运行，'
                  '按完整用户 ID 查无历史判定，本次所选窗口无历史运行记录。'
                  '保持模型、题卡、预选子集和可见域冻结。原生适配和窗口状态见上表。',''])
    overflow=rows['openviking','N']['status']['reader_overflow']
    if overflow:
        lines.extend([f'OpenViking 的 {overflow}/{len(code)} 道 coding N 题触发窗口限制。'
                      '价值分和综合分来自各场景的可判题，覆盖差异需结合全题分母和同题配对一起阅读。',''])
    if (root/'invalid-native/basic-memory-coding').is_dir():
        lines.extend(['Basic Memory 的首轮 coding 检出 MCP 项目绑定错误。表中使用合成笔记验证项目隔离后，'
                      '在新库完成的 coding 复测。其他域保留同一轮的正式结果。',''])
    if load(formal/'X2/daily-2/done.json',{}).get('settled_readback'):
        lines.extend(['X2 第二名 HaluMem 用户有一段的写入 agent 在部分提交后退出，适配器提前开始读取。'
                      '一条草案随后正常通过。表中使用同一隔离库达到审核终态后的读取与导出，'
                      '写入沿用原始摄取和实际审核用量。',''])
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(render(private_path(a.root)))


if __name__=='__main__':
    main()
