"""Execute the standalone console, worker and MCP outside the source directory."""
import json
from pathlib import Path
import subprocess
import tempfile
root=Path(__file__).resolve().parents[1]
binary=root/'dist/native/pi-trainer/pi-trainer'
report={}
with tempfile.TemporaryDirectory(prefix='pi-trainer-release-') as tmp:
    cwd=Path(tmp);state=cwd/'state'
    def call(*args):
        process=subprocess.run([str(binary),'--data-dir',str(state),*args],cwd=cwd,text=True,capture_output=True,timeout=45)
        if process.returncode:raise RuntimeError(process.stderr or process.stdout)
        return json.loads(process.stdout)
    project=call('create','--name','Frozen worker check','--target','hailo8l','--task','vision')
    call('import',project['id'],str(root/'examples/vision-smoke'))
    spec=cwd/'spec.json';spec.write_text('{}')
    job=call('run',project['id'],'export_dataset','--spec',str(spec),'--wait')
    assert job['status']=='succeeded',job
    assert Path(job['result']['archive_path']).is_file()
    report['standalone_detached_export_worker']=True
    messages=[{'jsonrpc':'2.0','id':1,'method':'initialize','params':{}},{'jsonrpc':'2.0','method':'notifications/initialized'}, {'jsonrpc':'2.0','id':2,'method':'tools/list'}, {'jsonrpc':'2.0','id':3,'method':'tools/call','params':{'name':'job_get','arguments':{'job_id':job['id']}}}]
    response=subprocess.run([str(binary),'--data-dir',str(state),'mcp'],input='\n'.join(json.dumps(m) for m in messages)+'\n',text=True,capture_output=True,cwd=cwd,timeout=10,check=True)
    rows=[json.loads(line) for line in response.stdout.splitlines()]
    assert len(rows)==3 and rows[0]['result']['serverInfo']['version']=='0.2.0'
    assert not rows[-1]['result']['isError']
    report['mcp_tool_count']=len(rows[1]['result']['tools']);report['mcp_job_read']=True
    report['controller_embedded_python']=call('doctor')['python_version']
report['platform']='macOS arm64';report['hardware_tested']=False
(root/'validation/release-smoke.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
