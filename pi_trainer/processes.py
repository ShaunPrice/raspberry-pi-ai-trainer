"""Bounded cross-platform subprocess execution for explicit workbench jobs."""
from __future__ import annotations
import os
import signal
import subprocess
import time
from pathlib import Path

class JobCancelled(RuntimeError):pass

def stop_process(process):
    if process.poll() is not None:return
    try:
        if os.name=='posix':os.killpg(process.pid,signal.SIGTERM)
        else:
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,timeout=10)
        process.wait(timeout=5)
    except (ProcessLookupError,subprocess.TimeoutExpired):
        if process.poll() is None:
            if os.name=='posix':os.killpg(process.pid,signal.SIGKILL)
            else:
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,timeout=10)
            process.wait(timeout=10)

def run_process(argv,cwd,log_path,cancelled,timeout,env=None):
    if not isinstance(argv,(list,tuple)) or not argv or any(not isinstance(x,str) or '\0' in x for x in argv):raise ValueError('Process requires an argument array')
    if not isinstance(timeout,(int,float)) or not 0<timeout<=7*86400:raise ValueError('Process timeout outside allowed range')
    if cancelled():raise JobCancelled('Job cancelled before process launch')
    log_path=Path(log_path);log_path.parent.mkdir(parents=True,exist_ok=True)
    environment=os.environ.copy()
    if env:environment.update(env)
    environment.setdefault('PYTHONUNBUFFERED','1')
    start=time.monotonic()
    kwargs={'start_new_session':True} if os.name=='posix' else {'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP}
    with log_path.open('ab') as log:
        p=subprocess.Popen(argv,cwd=str(cwd),env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,**kwargs)
        try:
            while p.poll() is None:
                if cancelled():raise JobCancelled('Job cancelled')
                if time.monotonic()-start>timeout:raise TimeoutError(f'Process exceeded {timeout}s deadline')
                if log_path.stat().st_size>32*1024*1024:raise RuntimeError('Process log exceeded 32 MiB limit')
                time.sleep(.1)
        finally:
            if p.poll() is None:stop_process(p)
    with log_path.open('rb') as f:
        f.seek(max(0,log_path.stat().st_size-1024*1024));output=f.read().decode('utf-8',errors='replace')
    result={'returncode':p.returncode,'stdout':output,'stderr':'','log_path':str(log_path),'elapsed_s':time.monotonic()-start}
    if p.returncode:raise RuntimeError(f'Process exited {p.returncode}. Log: {log_path}\n{output[-4000:]}')
    return result
