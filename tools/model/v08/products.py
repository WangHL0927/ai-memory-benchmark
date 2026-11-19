"""Cumulative product lifecycles and native read integrations for v0.8."""
from collections import Counter
from contextlib import closing, contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace
from urllib.parse import urlencode, urlsplit
import uuid

from tools.config import sha256_text
from tools.model.v08.native import HttpMCP, tool_list
from tools.model.v08.reader import Native
from tools.run import ai_memory as aim, basic_memory as bm, claude_mem as cm, openviking as ov, x2
from tools.run.proxy import ModelProxy
from tools.trial import now, write_json

PRODUCTS=('X2','ai-memory','basic-memory','claude-mem','openviking')


def text_for(root,item):
    text=(root/'samples'/item['id']/'transcript.md').read_text()
    if sha256_text(text)!=item['sha256']:
        raise ValueError('Frozen transcript hash mismatch')
    return text


def entries(records):
    return [dict(memory_id=f'm{i}',**row) for i,row in enumerate(records,1)]


def provider_usage(calls):
    result={}
    for kind in ('llm','embedding'):
        filtered=[c for c in calls if c['kind']==kind]
        usage={}
        for metric in ('prompt_tokens','completion_tokens','total_tokens'):
            amounts=[c.get('usage',{}).get(metric) if isinstance(c.get('usage'),dict) else None for c in filtered]
            usage[metric]=sum(amounts) if all(type(v) is int for v in amounts) else None
        result[kind]=dict(calls=len(filtered),tokens=usage)
    return result


class Product:
    def __init__(self, items, directory, config, *, reopen=False):
        self.items,self.directory,self.config=items,directory,config
        self.directory.mkdir(parents=True,mode=0o700,exist_ok=reopen)
        self.audit=directory/('reopen-audits-'+uuid.uuid4().hex if reopen else 'audits')
        self.audit.mkdir()
        self.projects=sorted({item['project'] for item in items})
        self.proxy=ModelProxy(config,self.audit,180)
        self.args=SimpleNamespace(timeout=3600,request_timeout=180,startup_timeout=240,v07=True)

    def project_path(self,item):
        cwd=self.directory/'workspaces'/item['project']
        cwd.mkdir(parents=True,exist_ok=True)
        return cwd

    def step(self,item,text):
        before=len(self.proxy.calls)
        started=time.monotonic()
        try:
            extra=self.ingest(item,text)
            calls=self.proxy.calls[before:]
            if any(c.get('status')!=200 for c in calls):
                raise RuntimeError('Writing model call failed or did not finish')
            outcome=dict(status='written',extra=extra)
        except (RuntimeError,ValueError,TimeoutError,OSError,KeyError) as exc:
            calls=self.proxy.calls[before:]
            outcome=dict(status='ingest_failure',error_type=type(exc).__name__,
                         issue=str(exc)[:300] if isinstance(exc,(RuntimeError,ValueError)) else None)
        manifest=dict(sample_id=item['id'],source_sha256=item['sha256'],usage=provider_usage(calls),
                      calls=calls,elapsed_seconds=round(time.monotonic()-started,3),end=now(),**outcome)
        write_json(self.audit/f"{item['id']}-ingest.json",manifest)
        print(item['id'],outcome['status'],manifest['usage']['llm']['calls'],flush=True)
        return manifest


class X2(Product):
    def __init__(self,*args):
        super().__init__(*args)
        source=Path(os.environ['X2_REPO']).resolve(strict=True)
        self.args.source=source
        self.args.node=shutil.which('node')
        self.args.port=int(os.environ['X2_PORT'])
        if self.args.port==18787:
            raise ValueError('Production port cannot be used')
        self.args.base=f'http://127.0.0.1:{self.args.port}'
        db_name=os.environ['X2_DATABASE_NAME']
        db_port=int(os.environ['X2_DATABASE_PORT'])
        url=x2.connection_string(Path(os.environ['X2_DATABASE_URL_FILE']),db_name,db_port)
        self.database=dict(PATH='/usr/bin:/bin:/usr/sbin:/sbin',HOME=str(self.directory),LANG='en_US.UTF-8',
                           DATABASE_URL=url,X2_EVAL_DB=db_name,X2_EVAL_DB_PORT=str(db_port))
        self.package=x2.fingerprint(source)
        required=os.environ['X2_EXPECTED_FINGERPRINT']
        if self.package['version']!=required:
            raise RuntimeError('X2 fingerprint differs from the frozen v0.7.3 source')
        self.proxy=ModelProxy(self.config,self.audit,180,track_phase=True,revise_model='deepseek-v4-pro',
                              revise_timeout=255,revise_json_keyword=True)

    def server_environment(self):
        return dict(self.database,TMPDIR=str(self.directory),HOST='127.0.0.1',PORT=str(self.args.port),
                    REVIEW_LLM_MODEL=self.config['model'],REVIEW_REVISE_LLM_MODEL='deepseek-v4-pro',
                    REVIEW_LLM_API_KEY='local-proxy',REVIEW_LLM_BASE_URL=self.proxy.url,REVIEW_LLM_TIMEOUT_MS='90000',
                    REVIEW_REVISE_LLM_TIMEOUT_MS='240000',REVIEW_INTERVAL_MS='1000')

    @contextmanager
    def open(self):
        x2.db(self.args,self.database,'prepare')
        self.tenant=x2.db(self.args,self.database,'tenant',tenant='x2-eval-'+uuid.uuid4().hex,projects=self.projects)
        install=self.directory/'client'
        env=self.server_environment()
        subprocess.run([self.args.node,str(self.args.source/os.environ['X2_INSTALLER_RELATIVE']),
                        '--project',str(install)],cwd=self.directory,env=env,check=True,capture_output=True,timeout=30)
        self.skill=(install/os.environ['X2_GENERATED_SKILL_RELATIVE']).read_text()
        with x2.isolated_server(self.args,self.directory,env):
            yield

    @contextmanager
    def reopen_for_read(self,draft_id):
        self.tenant=x2.db(self.args,self.database,'readback-key',draft_id=draft_id)
        try:
            with x2.isolated_server(self.args,self.directory,self.server_environment()):
                yield
        finally:
            x2.db(self.args,self.database,'revoke-readback-key',key_id=self.tenant['key_id'])

    def ingest(self,item,text):
        if x2.fingerprint(self.args.source)!=self.package:
            raise RuntimeError('X2 source changed during v0.8')
        a=self.audit/item['id']
        a.mkdir()
        client=x2.MCP(self.args.base,self.tenant['key'],a,120,session_id=item['id'])
        init=client.request('initialize',dict(protocolVersion='2025-03-26',capabilities={},
                                             clientInfo=dict(name='benchmark',version='1')))
        client.request('notifications/initialized',notification=True)
        instructions=init['instructions']
        if instructions.splitlines()[0] not in self.skill:
            raise RuntimeError('X2 intake version mismatch')
        before=client.upload_transcript(text,'before')
        state={}
        try:
            bm.agent_loop(self.proxy.url,self.config['model'],instructions+'\n\n'+self.skill,
                          x2.agent_input(text,[item['project']],item['id']),client.tools(),client.call,state,a,
                          12,2_000_000,180,time.monotonic()+900,allowed=x2.TOOLS)
        finally:
            after=client.upload_transcript(text,'after')
            deadline=time.monotonic()+900
            scope={k:self.tenant[k] for k in ('tenant_id','agent_id')}
            while x2.db(self.args,self.database,'pending',**scope):
                if time.monotonic()>deadline:
                    raise TimeoutError('X2 review did not reach terminal state')
                time.sleep(2)
        return dict(uploads=[before,after],agent=state)

    def export(self,group):
        raw=x2.db(self.args,self.database,'export',tenant_id=self.tenant['tenant_id'])
        if any(draft['state']=='pending' for draft in raw['drafts']):
            raise RuntimeError('X2 review has not settled')
        # All teams granted to this caller remain visible. Project affects native ranking.
        return entries(dict(title=r['title'],source=r['path'],scope=r['space_kind'],statement=r['body'])
                       for r in raw['pages'] if r['status']=='active')

    @contextmanager
    def reader(self,item,directory):
        client=HttpMCP(self.args.base,{'Authorization':'Bearer '+self.tenant['key']})
        client.begin()
        tools=tool_list(client.request,{'query','read'})
        def call(name,args):
            payload=dict(args,project=item['project']) if name=='query' else args
            return client.request('tools/call',dict(name=name,arguments=payload))
        # This pinned host integration only installs the intake Stop hook.
        yield Native(tools,call)


def clone_sql_store(source, destination, database):
    """Snapshot a quiescent native store with SQLite's backup API."""
    import sqlite3
    shutil.copytree(source,destination,ignore=shutil.ignore_patterns(database.name,database.name+'-wal',
        database.name+'-shm','*.pid','*.sock','*.lock'))
    relative=database.relative_to(source)
    with closing(sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)) as original:
        with closing(sqlite3.connect(destination/relative)) as clone:
            original.backup(clone)


def env_file(product, *path):
    base=Path(os.environ['BENCHMARK_PINNED_ENVS']).resolve(strict=True)
    return base/product/Path(*path)


class BasicMemory(Product):
    def __init__(self,*args,reopen=False):
        super().__init__(*args,reopen=reopen)
        self.reopen=reopen
        self.args.python=env_file('basic-memory','bin','python')
        prompts=list(self.args.python.parent.parent.glob('lib/python*/site-packages/basic_memory/data/skills/memory-capture/SKILL.md'))
        self.skill,=prompts
        from tools.config import sha256_file
        if sha256_file(self.skill)!=bm.PROMPT_SHA256:
            raise RuntimeError('Basic Memory skill differs from the pinned version')

    @contextmanager
    def open(self):
        with bm.isolated_mcp(self.args,self.directory,self.config,self.proxy,self.projects[0],projects=self.projects,
                             existing_store=self.reopen) as value:
            self.tools,self.call,self.deadline=value
            yield

    def ingest(self,item,text):
        a=self.audit/item['id']
        a.mkdir()
        project=item['project']
        state={}
        def call(name,args):
            if args.get('project',project)!=project or args.get('projects') or args.get('all_projects'):
                return bm.tool_error('This session may write and search its bound project only.')
            return self.call(name,args,scope=project)
        bm.agent_loop(self.proxy.url,self.config['model'],self.skill.read_text(),bm.agent_input(text,project,item['id']),
                      self.tools,call,state,a,12,2_000_000,180,min(time.monotonic()+900,self.deadline))
        return dict(agent=state)

    def export(self,group):
        import sqlite3
        project=next(item['project'] for item in self.items if item['group']==group)
        database=(self.directory/'config/memory.db').resolve()
        with closing(sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)) as db:
            rows=db.execute("""SELECT n.file_path,n.markdown_content FROM note_content n
                       JOIN project p ON n.project_id=p.id WHERE p.name=? ORDER BY n.file_path""",(project,))
            return entries(dict(source=path,statement=markdown) for path,markdown in rows)

    @contextmanager
    def reader(self,item,directory):
        names={'bm_search','bm_read','bm_list'}
        tools=[t for t in self.tools if t['function']['name'] in names]
        if not tools:
            raise RuntimeError('Pinned Basic Memory did not advertise search/read tools')
        def call(name,args):
            args={k:v for k,v in args.items() if k not in ('project','projects','all_projects')}
            try:
                return self.call(name,args,scope=item['project'])
            except RuntimeError as exc:
                return bm.tool_error(str(exc))
        # Native search/read/list do not append observations or change note text.
        yield Native(tools,call)


class AiMemory(Product):
    def __init__(self,*args):
        super().__init__(*args)
        self.binary=Path(os.environ['AI_MEMORY_347_BINARY']).resolve(strict=True)
        self.start_hook=Path(os.environ['AI_MEMORY_347_START_HOOK']).resolve(strict=True)
        self.seen=set()

    @contextmanager
    def open(self):
        with aim.isolated_server(self.binary,self.directory,self.config,self.proxy,120,
                                 project=self.projects[0],native_vectors=True) as (self.base,_):
            yield

    def ingest(self,item,text):
        cwd=self.project_path(item)
        session_id=str(uuid.uuid4())
        requests,_=aim.replay_requests(text,self.base,session_id,item['project'],cwd)
        with closing(aim.database(self.directory/'store')) as db:
            prior={r[0] for r in db.execute('select hex(id) from sessions')}
        acknowledgements=Counter()
        for offset in range(0,len(requests),256):
            part=requests[offset:offset+256]
            ack=aim.http(self.base,'/hook/batch',part,timeout=180)
            acknowledgements.update(aim.validate_ack(ack,len(part)))
        with closing(aim.database(self.directory/'store')) as db:
            sessions=[r for r in db.execute('''select hex(s.id) id,s.ended_at,p.name from sessions s
                 join projects p on p.id=s.project_id''') if r['id'] not in prior]
        if len(sessions)!=1 or sessions[0]['ended_at'] is None or sessions[0]['name']!=item['project']:
            raise RuntimeError('Native session is missing, not closed, or bound to the wrong project')
        sid=str(uuid.UUID(hex=sessions[0]['id']))
        aim.consolidate(self.base,sid,900,self.audit/f"{item['id']}-consolidation.json")
        return dict(ack=acknowledgements)

    def export(self,group):
        project=next(item['project'] for item in self.items if item['group']==group)
        store=self.directory/'store'
        records,structural,_=aim.export_pages(store)
        visible=[dict(source=r['path'],statement=r['body'],scope=r['project'],title=r['title'])
                 for r in records+structural if r['project'] in (project,'_global') and r['workspace']=='eval']
        with closing(aim.database(store)) as db:
            observations=db.execute("""SELECT hex(o.session_id) AS session, o.kind, o.title,
                        o.body, p.name AS scope FROM observations o
                        JOIN projects p ON o.project_id=p.id
                        JOIN workspaces w ON o.workspace_id=w.id
                        WHERE w.name='eval' AND p.name IN (?, '_global')
                        ORDER BY o.created_at,o.id""",(project,))
            visible.extend(dict(source=f"observation/{r['session']}/{r['kind']}",scope=r['scope'],
                                title=r['title'],statement=r['body']) for r in observations if r['body'])
            handoffs=db.execute("""SELECT h.summary, h.open_questions, h.next_steps, h.files_touched,
                        h.state, p.name AS scope FROM handoffs h JOIN projects p ON h.project_id=p.id
                        JOIN workspaces w ON h.workspace_id=w.id
                        WHERE w.name='eval' AND p.name IN (?, '_global') ORDER BY h.created_at""",(project,))
            visible.extend(dict(source='handoff',scope=r['scope'],statement=json.dumps(dict(r),ensure_ascii=False))
                           for r in handoffs)
        return entries(visible)

    @contextmanager
    def reader(self,item,directory):
        directory.mkdir(parents=True,exist_ok=True)
        store=self.directory/'store'
        clone_sql_store(store,directory/'store',store/'db/memory.sqlite')
        # Start the real session hook on a disposable copy, including its handoff behavior.
        with aim.isolated_server(self.binary,directory,self.config,self.proxy,120,
                                 project=item['project'],native_vectors=True) as (base,_):
            cwd=directory/'home'/item['project']
            cwd.mkdir()
            (cwd/'.ai-memory.toml').write_text(f'workspace = "eval"\nproject = "{item["project"]}"\n')
            env=dict(aim.child_environment(directory),AI_MEMORY_HOOK_URL=base)
            request=dict(cwd=str(cwd),session_id=str(uuid.uuid4()),hook_event_name='SessionStart',source='startup')
            value=subprocess.run(['/bin/sh',str(self.start_hook)],env=env,cwd=cwd,input=json.dumps(request),
                                 capture_output=True,text=True,check=True,timeout=30)
            packet=json.loads(value.stdout or '{}')
            start=packet.get('hookSpecificOutput',{}).get('additionalContext','')
            client=HttpMCP(base)
            client.begin()
            tools=tool_list(client.request,{'memory_query','memory_recent','memory_read_page',
                  'memory_read_session_observations','memory_briefing','memory_handoff_list'})
            def call(name,args):
                # Caller identity is fixed even when the model suggests a cross-project override.
                scope_overrides=('workspace','project','project_id','scopes')
                bound={key:value for key,value in args.items() if key not in scope_overrides}
                bound.update(workspace='eval',project=item['project'])
                if name=='memory_read_page' and bound.get('scope') not in (None,'global','project'):
                    bound.pop('scope')
                return client.request('tools/call',dict(name=name,arguments=bound))
            yield Native(tools,call,start)


class ClaudeMem(Product):
    def __init__(self,*args):
        super().__init__(*args)
        self.args.package=env_file('claude-mem','node_modules','claude-mem')
        self.args.bun=Path(shutil.which('bun') or '/nonexistent')
        if not self.args.bun.is_file():
            raise RuntimeError('Pinned Claude Memory runtime is missing')

    @contextmanager
    def open(self):
        with cm.isolated_worker(self.args,self.directory,self.config,self.proxy) as (self.base,self.process):
            yield

    def ingest(self,item,text):
        events,_=cm.replay_requests(text,'benchmark-'+uuid.uuid4().hex,item['project'],self.project_path(item))
        acks=[]
        deadline=time.monotonic()+3600
        first_call,first_error=len(self.proxy.calls),len(self.proxy.errors)
        class SegmentProxy:
            @property
            def calls(view):
                return self.proxy.calls[first_call:]
            @property
            def errors(view):
                return self.proxy.errors[first_error:]
        segment=SegmentProxy()
        for event in events:
            ack=cm.http(self.base,event['path'],event['body'],timeout=180)
            acks.append(dict(path=event['path'],ack=ack))
            cm.wait_idle(self.base,self.process,segment,deadline)
        write_json(self.audit/f"{item['id']}-events.json",acks)
        return dict(events=len(acks))

    def export(self,group):
        project=next(item['project'] for item in self.items if item['group']==group)
        with closing(cm.database(self.directory)) as db:
            rows=[dict(row,kind=kind) for kind,table in (('observation','observations'),('summary','session_summaries'))
                  for row in db.execute(f'SELECT * FROM {table} WHERE project=? ORDER BY rowid',(project,))]
        return entries(dict(source=f"{r['kind']}/{r['id']}",statement=cm.native_to_draft(r)['statement']) for r in rows)

    @contextmanager
    def reader(self,item,directory):
        directory.mkdir(parents=True,exist_ok=True)
        store=self.directory/'store'
        clone_sql_store(store,directory/'store',store/'claude-mem.db')
        with cm.isolated_worker(self.args,directory,self.config,self.proxy,existing_store=True) as (base,_):
            from urllib.request import urlopen
            query=urlencode(dict(projects=item['project'],platformSource='claude'))
            with urlopen(base+'/api/context/inject?'+query,timeout=45) as response:
                injection=response.read().decode()
            cwd=directory/'work'/item['project']
            cwd.mkdir()
            port=urlsplit(base).port
            env=cm.child_environment(directory,self.config,self.proxy,port)
            env.update(CLAUDE_MEM_WORKER_URL=base,CLAUDE_MEM_SERVER_URL=base)
            script=self.args.package/'plugin/scripts/mcp-server.cjs'
            with (directory/'read-mcp.log').open('w') as log:
                proc=subprocess.Popen([str(self.args.bun),'--preload',str(cm.GUARD),str(script)],cwd=cwd,env=env,
                         stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,start_new_session=True)
                client=bm.StdioMCP(proc,directory,time.monotonic()+900)
                try:
                    client.request('initialize',dict(protocolVersion='2025-03-26',capabilities={},
                                                      clientInfo=dict(name='benchmark',version='1')))
                    client.send('notifications/initialized')
                    tools=tool_list(client.request,{'search','timeline','get_observations','dumb_search'})
                    scoped={t['function']['name'] for t in tools if 'project' in
                            t['function']['parameters'].get('properties',{})}
                    def call(name,args):
                        return client.request('tools/call',dict(name=name,arguments=
                                              dict(args,project=item['project']) if name in scoped else args))
                    yield Native(tools,call,injection)
                finally:
                    proc.stdin.close()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.terminate()
                        proc.wait(timeout=5)


class OpenViking(Product):
    def __init__(self,*args):
        super().__init__(*args)
        self.python=env_file('openviking','bin','python')
        self.package=ov.product_metadata(self.python)
        self.sessions={p:[] for p in self.projects}
        self.user_roots={}

    def headers(self,project):
        return {'X-OpenViking-Account':'benchmark','X-OpenViking-User':project}

    @contextmanager
    def open(self):
        with ov.isolated_server(self.python,self.directory,self.config,self.proxy,180) as self.base:
            yield

    def ingest(self,item,text):
        project=item['project']
        headers=self.headers(project)
        sid='benchmark-'+uuid.uuid4().hex
        session=ov.result(self.base,'/api/v1/sessions',dict(session_id=sid,auto_commit_policy=None),headers=headers)
        root=session['uri'].split('/sessions/')[0]
        if any(root==seen for name,seen in self.user_roots.items() if name!=project):
            raise RuntimeError('OpenViking did not bind a distinct project namespace')
        self.user_roots[project]=root
        messages=ov.transcript_to_messages(text,include_header=True)
        for offset in range(0,len(messages),100):
            part=messages[offset:offset+100]
            added=ov.result(self.base,f'/api/v1/sessions/{sid}/messages/batch',dict(messages=part),headers=headers)
            if added['added']!=len(part):
                raise RuntimeError('OpenViking did not accept every message')
        commit=ov.result(self.base,f'/api/v1/sessions/{sid}/commit',
                         dict(keep_recent_count=0,enable_working_memory=False),headers=headers)
        ov.wait_commit(self.base,commit,time.monotonic()+1800,.5,headers=headers)
        queued=ov.result(self.base,'/api/v1/system/wait',{},timeout=180,headers=headers)
        if any(q.get('error_count',0) for q in queued.values()):
            raise RuntimeError('Native index queue reported an error')
        self.sessions[project].append(commit['archive_uri'].rstrip('/'))
        return dict(session_id=sid,archive=commit['archive_uri'])

    def export(self,group):
        project=next(item['project'] for item in self.items if item['group']==group)
        if project not in self.user_roots:
            return []
        root=self.user_roots[project]
        headers=self.headers(project)
        archives=self.sessions[project]
        files=ov.list_files(self.base,root,headers=headers)
        uris={x['uri'] for x in files if not x.get('isDir',x.get('is_dir',False)) and
                   '/memories/' in x['uri'] and not x['uri'].split('/')[-1].startswith('.')}
        all_records=[dict(uri=uri,content=ov.read_content(self.base,uri,headers=headers)) for uri in sorted(uris)]
        selected,_=ov.select_records(all_records,self.package['initial_templates'])
        for archive in archives:
            uri=archive+'/messages.jsonl'
            selected.append(dict(uri=uri,content=ov.read_content(self.base,uri,headers=headers)))
        return entries(dict(source=r['uri'],statement=r['content']) for r in selected)

    @contextmanager
    def reader(self,item,directory):
        client=HttpMCP(self.base,self.headers(item['project']))
        client.begin()
        tools=tool_list(client.request,{'find','read','list','abstract','overview'})
        def call(name,args):
            return client.request('tools/call',dict(name=name,arguments=args))
        yield Native(tools,call)
