import http.client
import io
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from pi_trainer.core import Store
from pi_trainer.server import Server
from pi_trainer.mcp import handle,serve

class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.store=Store(Path(self.temp.name).resolve()/'state')
    def tearDown(self):self.temp.cleanup()
    def test_mcp_handshake_tools_errors_and_notifications(self):
        result=handle(self.store,{'jsonrpc':'2.0','id':1,'method':'initialize','params':{}})
        self.assertEqual(result['result']['protocolVersion'],'2024-11-05')
        tools=handle(self.store,{'jsonrpc':'2.0','id':2,'method':'tools/list'})['result']['tools']
        self.assertIn('llm_recipe',[t['name'] for t in tools])
        self.assertIsNone(handle(self.store,{'jsonrpc':'2.0','method':'notifications/initialized'}))
        error=handle(self.store,{'jsonrpc':'2.0','id':3,'method':'tools/call','params':{'name':'create_project','arguments':{'name':'x','target':'hailo8','task':'llm'}}})
        self.assertTrue(error['result']['isError'])
        output=io.StringIO();serve(self.store,io.StringIO('bad\n{"jsonrpc":"2.0","method":"notifications/initialized"}\n'),output)
        self.assertEqual(len(output.getvalue().splitlines()),1)
        self.assertEqual(json.loads(output.getvalue())['error']['code'],-32700)
    def test_mcp_no_unknown_args(self):
        r=handle(self.store,{'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':'capabilities','arguments':{'command':'bad'}}})
        self.assertTrue(r['result']['isError'])
    def test_cli_and_mcp_share_state(self):
        cmd=[sys.executable,'-m','pi_trainer','--data-dir',str(self.store.root)]
        result=subprocess.run(cmd+['create','--name','Shared','--target','hailo10h','--task','llm'],capture_output=True,text=True,check=True)
        p=json.loads(result.stdout)
        r=handle(self.store,{'jsonrpc':'2.0','id':2,'method':'tools/call','params':{'name':'list_projects'}})
        self.assertEqual(json.loads(r['result']['content'][0]['text'])[0]['id'],p['id'])
    def test_http_roundtrip_and_origin_guards(self):
        server=Server(self.store,port=0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            def request(method,path,body=None,headers=None):
                connection=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)
                connection.request(method,path,json.dumps(body) if body is not None else None,headers or {})
                response=connection.getresponse();status=response.status;content=response.read();connection.close();return status,content
            self.assertEqual(request('GET','/')[0],200)
            self.assertEqual(request('GET','/../state.sqlite3')[0],404)
            self.assertEqual(request('GET','/api/state',headers={'Host':'attacker.test'})[0],403)
            data={'name':'Web project','target':'hailo8','task':'vision'}
            self.assertEqual(request('POST','/api/projects',data,{'Content-Type':'application/json','Origin':'https://evil.test'})[0],403)
            self.assertEqual(request('POST','/api/projects',data,{'Content-Type':'text/plain'})[0],415)
            code,body=request('POST','/api/projects',data,{'Content-Type':'application/json'})
            self.assertEqual(code,200);self.assertEqual(json.loads(body)['name'],'Web project')
        finally:server.shutdown();server.server_close();thread.join()
    def test_no_lan_binding(self):
        with self.assertRaises(ValueError):Server(self.store,host='0.0.0.0',port=0)
