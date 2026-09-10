"""Explicit registered Linux SSH workers for training, compilation and image copies."""
from __future__ import annotations
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import sys
import uuid
import zipfile
from pathlib import Path,PurePosixPath
from .processes import run_process,JobCancelled

OPTIONS=['-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10']

def safe_extract(path,destination,max_bytes=32*1024**3):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        entries=archive.infolist()
        if len(entries)>50000 or sum(i.file_size for i in entries)>max_bytes:raise ValueError('Worker archive exceeds limits')
        seen=set()
        for item in entries:
            name=item.filename;p=PurePosixPath(name)
            if not name or '\\' in name or p.is_absolute() or any(x in ('','..','.') for x in p.parts) or ':' in name or name in seen:raise ValueError('Unsafe worker archive path')
            seen.add(name);mode=(item.external_attr>>16)&0o170000
            if mode not in (0,stat.S_IFREG,stat.S_IFDIR):raise ValueError('Worker archive links/special files rejected')
            target=destination.joinpath(*p.parts)
            if destination.resolve() not in target.resolve().parents:raise ValueError('Worker archive escapes destination')
            if item.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(item) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,1024*1024)

def register_worker(store,name,host,python):
    from .workflows import settings_get,settings_set
    if not isinstance(name,str) or not 1<=len(name)<=80:raise ValueError('Worker name must be 1–80 characters')
    if not isinstance(host,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,252}',host):raise ValueError('Worker host must be an existing trusted SSH alias')
    if not isinstance(python,str) or not re.fullmatch(r'(?:/[A-Za-z0-9_.+/-]+|python[0-9.]*)',python):raise ValueError('Worker Python must be a Linux Python executable name or absolute path without spaces')
    workers=settings_get(store,'workers',[]);entry={'id':hashlib.sha256((host+'\0'+python).encode()).hexdigest()[:16],'name':name,'host':host,'python':python,'platform':'linux','status':'registered_not_verified'}
    settings_set(store,'workers',[x for x in workers if x['id']!=entry['id']]+[entry]);return entry

# Only paths naming checksum-verified downloaded output files/directories may be
# promoted into the local provider result. Original worker metadata is retained.
OUTPUT_PATH_FIELDS={'model_path','calibration_path','adapter_path','bundle_path',
                    'image_path','preprocessing_path','archive_path','output_dir',
                    'path','log_path','manifest_path','labels_path'}

def normalise_downloaded_result(report,target):
    target=Path(target).resolve()
    remote_root=PurePosixPath(report.get('output_root',''))
    if not remote_root.is_absolute() or '..' in remote_root.parts:
        raise ValueError('Remote output root is missing or unsafe')
    checks=report['files']
    def local_path(value):
        if not isinstance(value,str) or '\\' in value:return None
        remote=PurePosixPath(value)
        if '..' in remote.parts:return None
        try:relative=remote.relative_to(remote_root)
        except ValueError:return None
        name=(PurePosixPath('outputs')/relative).as_posix()
        local=target.joinpath(*PurePosixPath(name).parts)
        if name in checks and local.is_file():return str(local)
        prefix=name.rstrip('/')+'/'
        if local.is_dir() and any(key.startswith(prefix) for key in checks):return str(local)
        return None
    def convert(value):
        if isinstance(value,list):return [convert(v) for v in value]
        if not isinstance(value,dict):return value
        result={}
        for key,item in value.items():
            if key in OUTPUT_PATH_FIELDS:
                mapped=local_path(item)
                if mapped is not None:result[key]=mapped
            elif key=='artifact_paths':
                result[key]=[mapped for entry in item if (mapped:=local_path(entry)) is not None] if isinstance(item,list) else []
            else:result[key]=convert(item)
        return result
    original=report.get('result')
    if not isinstance(original,dict):raise ValueError('Remote provider result must be an object')
    return convert(original)

def run_remote(worker,kind,spec,workdir,emit,cancelled):
    if kind not in ['train','optimise','compile','image']:raise ValueError('Remote worker supports training, optimisation, compilation and image customisation')
    if cancelled():raise JobCancelled('Cancelled')
    host=worker['host'];python=worker['python'];remote_id=uuid.uuid4().hex;relative=f'.local/share/pi-trainer/worker-jobs/{remote_id}'
    payload=workdir/'remote-input.zip';remote_spec=dict(spec);remote_spec.pop('worker_id',None);remote_spec['python']=remote_spec.pop('provider_python',None) or python
    if not isinstance(remote_spec['python'],str) or not re.fullmatch(r'(?:/[A-Za-z0-9_.+/-]+|python[0-9.]*)',remote_spec['python']):raise ValueError('provider_python must name a Linux worker-local Python environment')
    # Only provider inputs are transferred. SDK/executable installations belong
    # to the registered worker; they are never guessed or silently installed.
    fields=['dataset_dir','dataset_path','data','validation_data','test_data','model_path','model','calibration_path','base_image','bundle_path','model_script_path','adapter_path']
    path_fields={};already={};total=0
    with zipfile.ZipFile(payload,'x',zipfile.ZIP_STORED,allowZip64=True) as archive:
        package=Path(__file__).parent
        for file in package.rglob('*.py'):
            if not file.is_symlink():archive.write(file,'app/pi_trainer/'+file.relative_to(package).as_posix())
        for key in fields:
            value=remote_spec.get(key)
            if not isinstance(value,str) or not value:continue
            path=Path(value).expanduser()
            if not path.exists():continue # a named model/vendor recipe may be worker-local
            if path.is_symlink():raise ValueError('Worker input symlinks are not supported')
            path=path.resolve()
            parent=next((p for p in already if p.is_dir() and p in path.parents),None)
            if parent:
                path_fields[key]=already[parent]+'/'+path.relative_to(parent).as_posix();continue
            if path in already:path_fields[key]=already[path];continue
            base='inputs/'+key+(path.suffix if path.is_file() else '')
            already[path]=base;path_fields[key]=base
            files=[path] if path.is_file() else sorted(path.rglob('*'))
            for file in files:
                if file.is_symlink():raise ValueError('Worker directory contains symlink')
                if file.is_dir():continue
                if not file.is_file():raise ValueError('Worker inputs must be regular files')
                total+=file.stat().st_size
                if total>32*1024**3:raise ValueError('Worker transfer exceeds 32 GiB')
                arc=base if path.is_file() else base+'/'+file.relative_to(path).as_posix()
                archive.write(file,arc)
        archive.writestr('request.json',json.dumps({'kind':kind,'spec':remote_spec,'path_fields':path_fields},allow_nan=False))
    def ssh(code,log,timeout=120,cancel=cancelled):
        return run_process(['ssh',*OPTIONS,host,shlex.join([python,'-c',code])],workdir,workdir/log,cancel,timeout)
    setup=f"from pathlib import Path;p=Path.home()/{relative!r};p.mkdir(parents=True,exist_ok=False);print(p)"
    emit({'stage':'worker_connect','worker':worker['name'],'bytes':payload.stat().st_size})
    ssh(setup,'remote-setup.log')
    run_started=False;remote_completed=False
    try:
        run_process(['scp',*OPTIONS,str(payload),host+':'+relative+'/input.zip'],workdir,workdir/'remote-upload.log',cancelled,1800)
        # Bootstrap is trusted application code; validate every archive entry
        # before extraction even though this client created the archive.
        bootstrap=f'''import os,sys,zipfile,stat
from pathlib import Path,PurePosixPath
p=Path.home()/{relative!r}
with zipfile.ZipFile(p/'input.zip') as z:
 for i in z.infolist():
  n=PurePosixPath(i.filename)
  if n.is_absolute() or '..' in n.parts or '\\\\' in i.filename or ':' in i.filename or stat.S_ISLNK(i.external_attr>>16): raise ValueError('Unsafe input path')
 z.extractall(p)
sys.path.insert(0,str(p/'app'))
from pi_trainer.remote_worker import run
run(p)
'''
        emit({'stage':'worker_running','worker':worker['name'],'remote_job':remote_id})
        run_started=True
        ssh(bootstrap,'remote-run.log',float(spec.get('timeout',86400))+600)
        remote_completed=True
        output=workdir/'remote-output.zip'
        run_process(['scp',*OPTIONS,host+':'+relative+'/output.zip',str(output)],workdir,workdir/'remote-download.log',cancelled,1800)
        target=workdir/'remote-result';safe_extract(output,target)
        report=json.loads((target/'result.json').read_text())
        if report.get('status')!='succeeded':raise RuntimeError('Remote worker failed: '+str(report.get('error')))
        checks=report['files']
        if not isinstance(checks,dict):raise ValueError('Invalid remote checksum manifest')
        actual={p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file() and p.relative_to(target).as_posix()!='result.json'}
        if set(checks)!=actual:raise ValueError('Remote checksum manifest does not match extracted files')
        for name,checksum in checks.items():
            parts=PurePosixPath(name)
            if not isinstance(name,str) or not name.startswith('outputs/') or '..' in parts.parts or '\\' in name or ':' in name or not isinstance(checksum,str) or not re.fullmatch('[0-9a-f]{64}',checksum):raise ValueError('Unsafe remote checksum entry')
            if (target/'outputs').resolve() not in (target/name).resolve().parents:raise ValueError('Remote artifact escapes output directory')
            path=target/name
            if not path.is_file():raise ValueError('Remote artifact is absent')
            with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if digest!=checksum:raise ValueError('Remote artifact checksum mismatch')
        normalised=normalise_downloaded_result(report,target)
        return {**normalised,'worker':worker,'remote_job':remote_id,'remote_result':report['result'],'download_dir':str(target),'artifacts':[{ 'path':str(target/name),'sha256':checksum} for name,checksum in checks.items()],
                'evidence':'Real SSH worker operation; device hardware validation remains separate'}
    except (JobCancelled,TimeoutError,RuntimeError) as original:
        if not run_started or remote_completed:raise
        try:
            ssh(f"from pathlib import Path;(Path.home()/{relative!r}/'cancel').touch()",'remote-cancel.log',15,lambda:False)
            detail='Remote cancellation signal delivered; inspect remote logs before retrying any external operation.'
        except Exception:detail='Remote cancellation could not be confirmed. Inspect registered worker before submitting another job.'
        if isinstance(original,JobCancelled):raise JobCancelled(detail) from original
        raise RuntimeError(str(original)+' '+detail) from original
