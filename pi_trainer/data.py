"""Versioned annotations, split materialisation and bounded data collection."""
from __future__ import annotations
import hashlib
import json
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from .core import bounded_file,encode,split_for
from .processes import run_process,JobCancelled

def select_dataset(store,project_id,dataset_id=None):
    choices=store.datasets(project_id)
    if dataset_id:
        choices=[d for d in choices if d['id']==dataset_id]
    if not choices:raise ValueError('Import a dataset before running this operation')
    return choices[-1]

def annotate(store,project_id,dataset_id,annotations_path):
    dataset=select_dataset(store,project_id,dataset_id)
    raw=bounded_file(Path(annotations_path),16*1024*1024);mapping=json.loads(raw)
    if not isinstance(mapping,dict):raise ValueError('Annotations must map record SHA256 or image relative path to {label, group}')
    known={i['sha256'] for i in dataset['items']}|{i['path'] for i in dataset['items']}
    if set(mapping)-known:raise ValueError('Annotation keys do not match dataset records')
    new=json.loads(encode(dataset));new['parent_dataset_id']=dataset_id;new['id']=uuid.uuid4().hex;new['created_at']=time.time()
    for item in new['items']:
        value=mapping.get(item['sha256'],mapping.get(item['path'],{}))
        if not isinstance(value,dict) or set(value)-{'label','group'}:raise ValueError('Supported annotation fields are label and group')
        if 'label' in value:
            label=value['label']
            if not isinstance(label,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',label):raise ValueError('Class labels must be 1–64 safe letters, digits, hyphens or underscores')
            item['label']=label
        if 'group' in value:
            group=value['group']
            if not isinstance(group,str) or not 1<=len(group)<=200:raise ValueError('Group must be a 1–200 character source/session identifier')
            item['group']=group
        if item.get('group'):item['split']=split_for(hashlib.sha256(item['group'].encode()).hexdigest())
        item['calibration']=item['split']=='train'
    new['split_counts']={s:sum(i['split']==s for i in new['items']) for s in ['train','validation','test']}
    new['annotations_sha256']=store.put_blob(raw);new['split_policy']='content hash with annotated groups kept together; calibration=train'
    new['sha256']=hashlib.sha256(encode({'files':new['files'],'items':new['items']}).encode()).hexdigest()
    with store.connection() as db:db.execute('INSERT INTO datasets VALUES(?,?,?)',(new['id'],project_id,encode(new)))
    return new

def materialise(store,project_id,dataset_id,out,require_splits=True):
    p=store.project(project_id);d=select_dataset(store,project_id,dataset_id)
    if require_splits and not all(d['split_counts'].values()):raise ValueError('Training requires nonempty training, validation and held-out test sets. Add data or correct group annotations.')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for split in ['train','validation','test']:(out/split).mkdir(exist_ok=True)
    blobs={}
    for file in d['files']:
        digest=file['sha256'];blob=store.blobs/digest
        content=bounded_file(blob)
        if hashlib.sha256(content).hexdigest()!=digest:raise ValueError('Dataset snapshot integrity failure')
        blobs[digest]=content
    classes=set();coverage={s:set() for s in ['train','validation','test']}
    if p['task']=='vision':
        for item in d['items']:
            label=item.get('label') or (Path(item['path']).parts[-2] if len(Path(item['path']).parts)>1 else None)
            if not label or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',label):raise ValueError('Vision training requires class folders or explicit safe class labels')
            classes.add(label);coverage[item['split']].add(label);folder=out/item['split']/label;folder.mkdir(exist_ok=True)
            extension=Path(item['path']).suffix.lower()
            (folder/(item['sha256']+extension)).write_bytes(blobs[item['file_sha256']])
        if len(classes)<2:raise ValueError('Classification training requires at least two classes')
        if require_splits and any(labels!=classes for labels in coverage.values()):raise ValueError('Every class requires training, validation and held-out test examples. Missing classes: '+str({s:sorted(classes-labels) for s,labels in coverage.items() if classes-labels}))
    else:
        decoded={digest:content.decode('utf-8').splitlines() for digest,content in blobs.items()}
        for split in ['train','validation','test']:
            with (out/(split+'.jsonl')).open('w',encoding='utf-8') as stream:
                for item in d['items']:
                    if item['split']==split:
                        rows=decoded[item['file_sha256']]
                        stream.write(encode(json.loads(rows[item['line']-1]))+'\n')
    manifest={'dataset_id':d['id'],'dataset_sha256':d['sha256'],'split_counts':d['split_counts'],'classes':sorted(classes),'task':p['task'],'target':p['target']}
    (out/'snapshot.json').write_text(encode(manifest))
    return {'dataset_dir':str(out),'data':str(out/'train.jsonl'),'validation_data':str(out/'validation.jsonl'),'test_data':str(out/'test.jsonl'),**manifest}

def export_dataset(store,project_id,spec,workdir,emit,cancelled):
    if cancelled():raise JobCancelled('Cancelled')
    result=materialise(store,project_id,spec.get('dataset_id'),workdir/'dataset',False)
    emit({'stage':'dataset_export','message':'Verified immutable files materialised'})
    archive=shutil.make_archive(str(workdir/'dataset'),'zip',workdir/'dataset')
    return {**result,'archive_path':archive,'sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest()}

def capture(spec,workdir,emit,cancelled):
    seconds=float(spec.get('seconds',5));fps=float(spec.get('fps',1));index=spec.get('camera_index',0)
    if not 0<seconds<=300 or not 0<fps<=30 or seconds*fps>1000:raise ValueError('Capture limited to 300 seconds and 1,000 frames')
    if type(index) is not int or not 0<=index<=16:raise ValueError('Camera index must be between 0 and 16')
    worker=Path(__file__).with_name('capture_worker.py')
    output=workdir/'capture';output.mkdir()
    from .runtime import compute_python
    args=[compute_python(spec),str(worker),'--output',str(output),'--seconds',str(seconds),'--fps',str(fps),'--camera',str(index)]
    if spec.get('video_path'):args+=['--video',str(Path(spec['video_path']).resolve())]
    run_process(args,workdir,workdir/'capture.log',cancelled,seconds+120)
    metadata=json.loads((output/'capture.json').read_text());emit({'stage':'capture','frames':metadata['frames']})
    return {'capture_dir':str(output),**metadata,'next_step':'Import capture folder and annotate classes/groups before training'}
