"""Durable local jobs, one worker lease and explicit cancellation across interfaces."""
from __future__ import annotations
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from .processes import JobCancelled

TERMINAL={'succeeded','failed','cancelled','interrupted'}
KINDS={'train','compile','package','image','deploy','benchmark','rollback','capture','export_dataset','optimise'}

def initialise(store):
    with store.connection() as db:
        db.executescript('''CREATE TABLE IF NOT EXISTS workflow_jobs(id TEXT PRIMARY KEY, status TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS worker_lock(id INTEGER PRIMARY KEY CHECK(id=1), owner TEXT, pid INTEGER, heartbeat REAL);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, body TEXT NOT NULL);''')

def get(store,job_id):
    initialise(store)
    with store.connection() as db:r=db.execute('SELECT body FROM workflow_jobs WHERE id=?',(job_id,)).fetchone()
    if not r:raise ValueError('Unknown workflow job')
    return json.loads(r[0])

def list_all(store):
    initialise(store)
    with store.connection() as db:return [json.loads(r[0]) for r in db.execute('SELECT body FROM workflow_jobs ORDER BY rowid DESC LIMIT 100')]

def update(store,job_id,**changes):
    with store.connection() as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT body FROM workflow_jobs WHERE id=?',(job_id,)).fetchone()
        if not row:raise ValueError('Unknown workflow job')
        job=json.loads(row[0]);job.update(changes);job['updated_at']=time.time()
        db.execute('UPDATE workflow_jobs SET status=?,body=? WHERE id=?',(job['status'],json.dumps(job,allow_nan=False),job_id))
    return job

def enqueue(store,project_id,kind,spec):
    initialise(store);project=store.project(project_id)
    if kind not in KINDS:raise ValueError('Unknown workflow kind')
    if not isinstance(spec,dict):raise ValueError('Job spec must be an object')
    if len(json.dumps(spec,allow_nan=False))>1024*1024:raise ValueError('Job specification too large')
    if spec.get('target',project['target'])!=project['target'] or spec.get('task',project['task'])!=project['task']:raise ValueError('Job target/task must match project')
    spec={**spec,'target':project['target'],'task':project['task']}
    if kind in {'train','optimise','export_dataset'}:
        from .data import select_dataset
        dataset=select_dataset(store,project_id,spec.get('dataset_id'))
        spec['dataset_id']=dataset['id']
        spec['dataset_sha256']=dataset['sha256']
    job={'id':uuid.uuid4().hex,'project_id':project_id,'kind':kind,'spec':spec,'status':'queued','cancel_requested':False,'created_at':time.time(),'updated_at':time.time(),'progress':{},'result':None,'error':None}
    with store.connection() as db:db.execute('INSERT INTO workflow_jobs VALUES (?,?,?)',(job['id'],job['status'],json.dumps(job)))
    spawn_worker(store)
    return job

def spawn_worker(store):
    logdir=store.root/'logs';logdir.mkdir(exist_ok=True)
    environment=os.environ.copy();source=str(Path(__file__).resolve().parent.parent)
    environment['PYTHONPATH']=source+os.pathsep+environment.get('PYTHONPATH','')
    kwargs={'start_new_session':True} if os.name=='posix' else {'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
    with (logdir/'worker.log').open('ab') as log:
        argv=[sys.executable,'--worker',str(store.root)] if getattr(sys,'frozen',False) else [sys.executable,'-m','pi_trainer.worker','--data-dir',str(store.root)]
        process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=log,env=environment,**kwargs)
        threading.Thread(target=process.wait,daemon=True).start()

def cancel(store,job_id):
    initialise(store)
    with store.connection() as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT body FROM workflow_jobs WHERE id=?',(job_id,)).fetchone()
        if not row:raise ValueError('Unknown workflow job')
        job=json.loads(row[0])
        if job['status'] in TERMINAL:return job
        job['cancel_requested']=True;job['updated_at']=time.time()
        if job['status']=='queued':job['status']='cancelled'
        db.execute('UPDATE workflow_jobs SET status=?,body=? WHERE id=?',(job['status'],json.dumps(job),job_id))
    return job

def alive(pid):
    if type(pid) is not int or pid<=0:return False
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:return ctypes.get_last_error()==5 # access denied is not proof of exit
        try:
            code=wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
        finally:kernel.CloseHandle(handle)
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
    except PermissionError:return True

def acquire(store,owner):
    initialise(store)
    with store.connection() as db:
        db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT owner,pid,heartbeat FROM worker_lock WHERE id=1').fetchone()
        if row and alive(row[1]):return False
        # A crashed worker is never silently retried, especially for device writes.
        for jid,body in db.execute("SELECT id,body FROM workflow_jobs WHERE status='running'").fetchall():
            j=json.loads(body);j.update(status='interrupted',error='Worker exited. Inspect logs and external state before explicitly submitting another job.',updated_at=time.time())
            db.execute('UPDATE workflow_jobs SET status=?,body=? WHERE id=?',('interrupted',json.dumps(j),jid))
        db.execute('INSERT OR REPLACE INTO worker_lock VALUES(1,?,?,?)',(owner,os.getpid(),time.time()))
    return True

def worker_loop(store):
    owner=uuid.uuid4().hex
    if not acquire(store,owner):return
    stop=threading.Event()
    def heartbeat():
        while not stop.wait(2):
            with store.connection() as db:db.execute('UPDATE worker_lock SET heartbeat=? WHERE owner=?',(time.time(),owner))
    threading.Thread(target=heartbeat,daemon=True).start()
    try:
        while True:
            with store.connection() as db:
                db.execute('BEGIN IMMEDIATE')
                row=db.execute("SELECT id,body FROM workflow_jobs WHERE status='queued' ORDER BY rowid LIMIT 1").fetchone()
                if not row:
                    # Release while holding the same database transaction used to see
                    # an empty queue so enqueue/spawn cannot lose a wake-up.
                    db.execute('DELETE FROM worker_lock WHERE owner=?',(owner,));break
                job=json.loads(row[1]);job.update(status='running',started_at=time.time(),updated_at=time.time())
                db.execute('UPDATE workflow_jobs SET status=?,body=? WHERE id=?',('running',json.dumps(job),job['id']))
            workdir=store.root/'runs'/job['id'];workdir.mkdir(parents=True,exist_ok=False)
            def cancelled():return bool(get(store,job['id'])['cancel_requested'])
            def emit(value):
                value=value if isinstance(value,dict) else {'message':str(value)}
                update(store,job['id'],progress=value)
                with (workdir/'events.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),**value},allow_nan=False)+'\n')
            try:
                from .workflows import execute
                result=execute(store,job,workdir,emit,cancelled)
                if cancelled():raise JobCancelled('Cancelled after operation; inspect artifacts and device state')
                json.dumps(result,allow_nan=False)
                update(store,job['id'],status='succeeded',result=result,finished_at=time.time(),workdir=str(workdir))
            except JobCancelled as exc:update(store,job['id'],status='cancelled',error=str(exc),finished_at=time.time(),workdir=str(workdir))
            except Exception as exc:update(store,job['id'],status='failed',error=f'{type(exc).__name__}: {exc}',finished_at=time.time(),workdir=str(workdir))
    finally:
        stop.set()
        with store.connection() as db:db.execute('DELETE FROM worker_lock WHERE owner=?',(owner,))

def job_logs(store,job_id):
    job=get(store,job_id);workdir=store.root/'runs'/job['id'];logs={}
    if workdir.exists():
        for file in sorted(workdir.rglob('*.log'))[:20]:
            if file.is_symlink():continue
            with file.open('rb') as f:f.seek(max(0,file.stat().st_size-64000));logs[str(file.relative_to(workdir))]=f.read().decode('utf-8',errors='replace')
    return {'job':job,'logs':logs}
