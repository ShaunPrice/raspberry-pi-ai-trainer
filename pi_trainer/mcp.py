"""Local newline-delimited JSON-RPC stdio MCP preparation server."""
from __future__ import annotations
import json
import sys
from .operations import dispatch,schemas

PROTOCOL='2024-11-05'

def handle(store,message):
    if not isinstance(message,dict) or message.get('jsonrpc')!='2.0' or not isinstance(message.get('method'),str):
        return {'jsonrpc':'2.0','id':None,'error':{'code':-32600,'message':'Invalid Request'}}
    notification='id' not in message
    request_id=message.get('id')
    if isinstance(request_id,(dict,list,bool)):
        return {'jsonrpc':'2.0','id':None,'error':{'code':-32600,'message':'Invalid request ID'}}
    if notification:return None
    base={'jsonrpc':'2.0','id':request_id}
    method=message['method'];params=message.get('params',{})
    if not isinstance(params,dict):return {**base,'error':{'code':-32602,'message':'params must be an object'}}
    if method=='initialize':return {**base,'result':{'protocolVersion':PROTOCOL,'capabilities':{'tools':{'listChanged':False}},'serverInfo':{'name':'pi-trainer','version':'0.2.0'},'instructions':'Local training/build/deployment workbench. run_job performs explicit operations; inspect status and evidence. SDKs and enrolled hardware are required for their stages.'}}
    if method=='ping':return {**base,'result':{}}
    if method=='tools/list':return {**base,'result':{'tools':schemas()}}
    if method=='tools/call':
        try:
            result=dispatch(store,params.get('name'),params.get('arguments',{}))
            content={'content':[{'type':'text','text':json.dumps(result,allow_nan=False)}],'isError':False}
        except (ValueError,TypeError,OSError,KeyError) as exc:
            content={'content':[{'type':'text','text':str(exc)}],'isError':True}
        except Exception:
            content={'content':[{'type':'text','text':'Internal operation error'}],'isError':True}
        return {**base,'result':content}
    return {**base,'error':{'code':-32601,'message':'Method not found'}}

def serve(store,stdin=None,stdout=None):
    stdin=stdin or sys.stdin;stdout=stdout or sys.stdout
    while True:
        line=stdin.readline(1024*1024+1)
        if not line:break
        if len(line)>1024*1024:
            result={'jsonrpc':'2.0','id':None,'error':{'code':-32700,'message':'Message exceeds 1 MiB'}}
            while not line.endswith('\n'):
                line=stdin.readline(1024*1024)
                if not line:break
        else:
            try:result=handle(store,json.loads(line))
            except (ValueError,TypeError):result={'jsonrpc':'2.0','id':None,'error':{'code':-32700,'message':'Parse error'}}
        if result is not None:stdout.write(json.dumps(result,allow_nan=False)+'\n');stdout.flush()
