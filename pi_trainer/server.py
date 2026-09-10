"""Loopback-only HTTP interface. No CORS, remote auth or generic file server."""
from __future__ import annotations
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from .operations import dispatch

ROUTES={'projects':'create_project','datasets':'import_dataset','plan':'plan','bundle':'bundle','llm-recipe':'llm_recipe','score':'score','estimate':'estimate'}
MAX_BODY=17*1024*1024

class Server(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,store,host='127.0.0.1',port=8765):
        if host!='127.0.0.1':raise ValueError('Prototype HTTP server must bind to 127.0.0.1')
        self.store=store
        super().__init__((host,port),Handler)

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def valid_request(self):
        port=self.server.server_port
        allowed={f'127.0.0.1:{port}',f'localhost:{port}'}
        if self.headers.get('Host') not in allowed:return False
        if self.headers.get('Sec-Fetch-Site')=='cross-site':return False
        origin=self.headers.get('Origin')
        if origin is not None and origin!='http://'+self.headers.get('Host'):return False
        return True
    def reply(self,status,body,kind='application/json'):
        data=json.dumps(body,allow_nan=False).encode() if kind=='application/json' else body
        self.send_response(status);self.send_header('Content-Type',kind+'; charset=utf-8')
        self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff');self.send_header('X-Frame-Options','DENY')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.end_headers();self.wfile.write(data)
    def do_GET(self):
        if not self.valid_request():return self.reply(403,{'error':'Host or origin rejected'})
        path=urlsplit(self.path).path
        if path=='/api/tools':
            from .operations import schemas
            return self.reply(200,{'tools':schemas()})
        if path=='/favicon.ico':return self.reply(204,b'', 'image/x-icon')
        if path=='/api/state':return self.reply(200,{'projects':self.server.store.list_projects(),'jobs':self.server.store.list_jobs(),'capabilities':self.server.store.capabilities(),'workflow_jobs':self.server.store.workflow_jobs(),'devices':self.server.store.devices(),'workers':self.server.store.workers()})
        if path=='/static/workflows.js':return self.reply(200,Path(__file__).with_name('static').joinpath('workflows.js').read_bytes(),'application/javascript')
        if path in ['/','/index.html']:
            return self.reply(200,Path(__file__).with_name('static').joinpath('index.html').read_bytes(),'text/html')
        return self.reply(404,{'error':'Not found'})
    def do_POST(self):
        if not self.valid_request():return self.reply(403,{'error':'Host or origin rejected'})
        if self.headers.get('Content-Type','').split(';')[0].strip()!='application/json':return self.reply(415,{'error':'application/json required'})
        if self.headers.get('Transfer-Encoding'):return self.reply(400,{'error':'Transfer encoding not supported'})
        try:
            length=int(self.headers.get('Content-Length','-1'))
            if not 0<=length<=MAX_BODY:return self.reply(413,{'error':'Request body must be <=17 MiB'})
            self.connection.settimeout(10)
            body=json.loads(self.rfile.read(length))
            path=urlsplit(self.path).path
            name=ROUTES.get(path.removeprefix('/api/')) if path.startswith('/api/') else None
            if path.startswith('/api/tools/'):
                name=path.removeprefix('/api/tools/')
            if name is None:return self.reply(404,{'error':'Not found'})
            result=dispatch(self.server.store,name,body)
            return self.reply(200,result)
        except (ValueError,TypeError,OSError,KeyError) as exc:return self.reply(400,{'error':str(exc)})
        except Exception:return self.reply(500,{'error':'Internal operation error'})

def serve(store,host='127.0.0.1',port=8765):
    server=Server(store,host,port)
    print(f'Pi Trainer: http://127.0.0.1:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
