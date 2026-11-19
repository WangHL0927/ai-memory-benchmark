"""Minimal streamable HTTP MCP client for isolated product integrations."""
import json
from urllib.error import HTTPError
from urllib.request import build_opener, ProxyHandler, Request

from tools.run.x2 import NoRedirect


class HttpMCP:
    def __init__(self, base, headers=()):
        self.base=base
        self.headers={'Accept':'application/json, text/event-stream','Content-Type':'application/json',**dict(headers)}
        self.opener=build_opener(ProxyHandler({}),NoRedirect())
        self.seq=0

    def request(self, method, params=None, *, notify=False):
        self.seq+=1
        packet=dict(jsonrpc='2.0',method=method)
        if not notify:
            packet['id']=self.seq
        if params is not None:
            packet['params']=params
        req=Request(self.base+'/mcp',data=json.dumps(packet,ensure_ascii=False).encode(),headers=self.headers)
        try:
            with self.opener.open(req,timeout=90) as response:
                sid=response.headers.get('Mcp-Session-Id')
                if sid:
                    self.headers['Mcp-Session-Id']=sid
                if notify:
                    return None
                if 'text/event-stream' in response.headers.get('Content-Type',''):
                    reply=None
                    for line in response:
                        if line.startswith(b'data:'):
                            value=json.loads(line[5:])
                            if value.get('id')==self.seq:
                                reply=value
                                break
                else:
                    reply=json.load(response)
        except HTTPError as exc:
            with exc:
                if method=='tools/call':
                    return dict(isError=True,http_status=exc.code)
            raise RuntimeError(f'MCP HTTP {exc.code}') from None
        if reply and reply.get('id')==self.seq:
            if 'result' in reply:
                return reply['result']
            if method=='tools/call':
                return dict(isError=True,error=reply.get('error'))
        raise RuntimeError('MCP response is missing or invalid')

    def begin(self):
        init=self.request('initialize',dict(protocolVersion='2025-03-26',capabilities={},
                                           clientInfo=dict(name='benchmark',version='1')))
        self.headers['MCP-Protocol-Version']=init['protocolVersion']
        self.request('notifications/initialized',notify=True)
        return init


def tool_list(request, names):
    all_tools, cursor = [],None
    while True:
        result=request('tools/list',{'cursor':cursor} if cursor else {})
        all_tools.extend(result['tools'])
        cursor=result.get('nextCursor')
        if not cursor:
            break
    selected=[dict(type='function',function=dict(name=t['name'],description=t.get('description',''),
                   parameters=t['inputSchema'])) for t in all_tools if t['name'] in names]
    if not selected:
        raise RuntimeError('No configured native read tools are advertised')
    return selected
