"""Workflow dispatch shared by the durable worker and all user interfaces."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import sys
from pathlib import Path
from . import jobs
from .data import materialise,export_dataset,capture
from .processes import JobCancelled


def doctor():
    return {'system':platform.system(),'machine':platform.machine(),'python':sys.executable,'python_version':platform.python_version(),
            'executables':{x:shutil.which(x) for x in ['docker','ssh','scp','guestfish','debugfs','hailortcli','soup','rpi-imager']},
            'python_modules':{x:importlib.util.find_spec(x) is not None for x in ['torch','torchvision','onnx','transformers','peft','hailo_sdk_client','cv2']},
            'evidence':'Installed tool discovery only. Does not prove backend or device operation.',
            'training_location':'off-Pi only'}

def settings_get(store,key,default):
    jobs.initialise(store)
    with store.connection() as db:r=db.execute('SELECT body FROM settings WHERE key=?',(key,)).fetchone()
    return json.loads(r[0]) if r else default

def settings_set(store,key,value):
    jobs.initialise(store)
    with store.connection() as db:db.execute('INSERT OR REPLACE INTO settings VALUES(?,?)',(key,json.dumps(value)))
    return value

def enroll(store,name,host):
    if not isinstance(name,str) or not 1<=len(name)<=80:raise ValueError('Device name must have 1–80 characters')
    if not isinstance(host,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,252}',host):raise ValueError('Use an existing trusted SSH alias; configure user/port in SSH config')
    devices=settings_get(store,'devices',[])
    entry={'id':hashlib.sha256(host.encode()).hexdigest()[:16],'name':name,'host':host,'status':'enrolled_unprobed'}
    devices=[x for x in devices if x['id']!=entry['id']]+[entry]
    settings_set(store,'devices',devices);return entry

def execute(store,job,workdir,emit,cancelled):
    spec=dict(job['spec']);kind=job['kind'];project_id=job['project_id']
    if cancelled():raise JobCancelled('Cancelled')
    emit({'stage':'preflight','kind':kind,'message':'Validating operation inputs'})
    if kind in ['train','optimise']:
        dataset=materialise(store,project_id,spec.get('dataset_id'),workdir/'dataset')
        spec.update({k:dataset[k] for k in ['dataset_dir','data','validation_data','test_data']})
        spec['dataset_path']=spec['data']
    if spec.get('worker_id'):
        from .remote import run_remote
        workers=settings_get(store,'workers',[])
        worker=next((w for w in workers if w['id']==spec['worker_id']),None)
        if not worker:raise ValueError('Unknown registered worker')
        return run_remote(worker,kind,spec,workdir,emit,cancelled)
    if kind=='optimise':
        from .optimisation import optimise
        return optimise(spec,workdir,emit,cancelled)
    if kind=='train':
        from .training import run_training
        return run_training(spec,workdir,emit,cancelled)
    if kind=='compile':
        from .compiler import run_compile
        return run_compile(spec,workdir,emit,cancelled)
    if kind=='package':
        from .deployment import create_bundle
        return create_bundle(spec,workdir,emit,cancelled)
    if kind=='image':
        from .images import customise_image
        return customise_image(spec,workdir,emit,cancelled)
    if kind in ['deploy','benchmark','rollback']:
        from .deployment import deploy,test_device,rollback
        devices=settings_get(store,'devices',[])
        device=next((d for d in devices if d['id']==spec.get('device_id')),None)
        if not device:raise ValueError('Enroll and select a device before network operations')
        spec['host']=device['host']
        if kind=='deploy':
            from .deployment import validate_bundle
            manifest=validate_bundle(spec.get('bundle_path',''))
            if manifest['target']!=spec['target']:raise ValueError('Deployment bundle target differs from project')
        return {'deploy':deploy,'benchmark':test_device,'rollback':rollback}[kind](spec,workdir,emit,cancelled)
    if kind=='capture':return capture(spec,workdir,emit,cancelled)
    if kind=='export_dataset':return export_dataset(store,project_id,spec,workdir,emit,cancelled)
    raise ValueError('Unknown operation')
