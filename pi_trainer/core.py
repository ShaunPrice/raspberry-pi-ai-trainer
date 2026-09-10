"""Shared persistent preparation operations. Never simulates training or compilation."""
from __future__ import annotations
import hashlib
from contextlib import contextmanager
import json
import os
import shutil
import sqlite3
import stat
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

TARGETS={'hailo8l':['vision'],'hailo8':['vision'],'hailo10h':['vision','llm']}
MAX_BYTES=512*1024*1024
MAX_FILES=10000

def encode(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)
def uid():return uuid.uuid4().hex

def split_for(digest):
    value=int(digest[:8],16)%100
    return 'train' if value<80 else 'validation' if value<90 else 'test'

def bounded_file(path,limit=MAX_BYTES):
    path=Path(path)
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):raise ValueError('Only regular files are supported; symlinks are rejected')
    if path.stat().st_size>limit:raise ValueError('File exceeds size limit')
    # O_NOFOLLOW closes the final-component symlink race on Unix.
    fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
    with os.fdopen(fd,'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):raise ValueError('Not a regular file')
        data=stream.read(limit+1)
    if len(data)>limit:raise ValueError('File exceeds size limit')
    return data

class Store:
    def __init__(self,root):
        self.root=Path(root).expanduser().resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.blobs=self.root/'blobs';self.blobs.mkdir(exist_ok=True)
        self.exports=self.root/'exports';self.exports.mkdir(exist_ok=True)
        self.db=self.root/'state.sqlite3'
        with self.connection() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS projects(id TEXT PRIMARY KEY,body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY,project_id TEXT NOT NULL,body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,body TEXT NOT NULL);''')
    @contextmanager
    def connection(self):
        db=sqlite3.connect(self.db,timeout=30)
        try:
            db.execute('PRAGMA journal_mode=WAL')
            with db:yield db
        finally:db.close()
    def capabilities(self):
        return {'schema':'pi-trainer/capabilities/v1','targets':TARGETS,
                'available':['projects','dataset_snapshots','build_plans','preparation_bundles','llm_recipe','quality_scoring','memory_estimate'],
                'implemented_providers':['off_pi_training','onnx_export','hailo_dfc_compile','vendor_genai_compile','model_packaging','ssh_deployment','device_benchmarks','rollback','existing_image_customisation','data_capture','remote_linux_workers','optimisation'],
                'not_implemented':['direct_raw_disk_writer'],
                'media_writing':'Raspberry Pi Imager handoff',
                'backend_status':'Installed framework/SDK/tool discovery available via doctor; physical validation remains separate',
                'hardware':{'hailo8l':'user_owned_not_tested','hailo8':'user_owned_not_tested','hailo10h':'testing_deferred'},
                'llm_optimise':'MIT recipe, quality and memory utilities reused; optional off-Pi training providers execute in isolated processes',
                'interfaces':['web','native_tk','cli','mcp_stdio']}
    def list_projects(self):
        with self.connection() as db:return [json.loads(r[0]) for r in db.execute('SELECT body FROM projects ORDER BY rowid')]
    def create_project(self,name,target,task):
        if not isinstance(name,str) or not 1<=len(name.strip())<=120:raise ValueError('Project name must contain 1–120 characters')
        if not isinstance(target,str) or target not in TARGETS:raise ValueError('Unknown Hailo target')
        if task not in TARGETS[target]:raise ValueError('LLM projects require Hailo-10H; 8/8L support vision projects')
        p={'id':uid(),'name':name.strip(),'target':target,'task':task,'created_at':time.time()}
        with self.connection() as db:db.execute('INSERT INTO projects VALUES (?,?)',(p['id'],encode(p)))
        return p
    def project(self,project_id):
        if not isinstance(project_id,str):raise ValueError('project_id must be a string')
        with self.connection() as db:r=db.execute('SELECT body FROM projects WHERE id=?',(project_id,)).fetchone()
        if not r:raise ValueError('Unknown project ID')
        return json.loads(r[0])
    def datasets(self,project_id):
        self.project(project_id)
        with self.connection() as db:return [json.loads(r[0]) for r in db.execute('SELECT body FROM datasets WHERE project_id=? ORDER BY rowid',(project_id,))]
    def list_jobs(self):
        with self.connection() as db:return [json.loads(r[0]) for r in db.execute('SELECT body FROM jobs ORDER BY rowid DESC LIMIT 100')]
    def record_job(self,kind,project_id,result,status='completed'):
        j={'id':uid(),'kind':kind,'project_id':project_id,'status':status,'created_at':time.time(),'result':result}
        with self.connection() as db:db.execute('INSERT INTO jobs VALUES (?,?)',(j['id'],encode(j)))
        return j
    def put_blob(self,data):
        digest=hashlib.sha256(data).hexdigest();dest=self.blobs/digest
        if dest.exists():
            if hashlib.sha256(bounded_file(dest)).hexdigest()!=digest:raise ValueError('Stored blob integrity failure')
            return digest
        fd,tmp=tempfile.mkstemp(dir=self.blobs)
        try:
            with os.fdopen(fd,'wb') as f:f.write(data)
            os.replace(tmp,dest)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
        return digest
    def import_dataset(self,project_id,path):
        project=self.project(project_id)
        if not isinstance(path,str) or not path.strip():raise ValueError('Dataset path is required')
        raw=Path(path).expanduser()
        if any(p.is_symlink() for p in [raw,*raw.parents]):raise ValueError('Symlink paths are not accepted')
        source=raw.resolve()
        managed_upload=source.parent==self.root/'uploads'
        managed_capture=self.root/'runs' in source.parents and source.name=='capture'
        if (source==self.root or self.root in source.parents or source in self.root.parents) and not (managed_upload or managed_capture):raise ValueError('Choose an external dataset, a managed capture folder, or a browser upload')
        if not source.exists():raise ValueError('Dataset path does not exist')
        paths=[]
        if source.is_dir():
            for base,dirs,files in os.walk(source,followlinks=False):
                for name in dirs+files:
                    if (Path(base)/name).is_symlink():raise ValueError('Dataset contains a symlink')
                for name in sorted(files):
                    paths.append(Path(base)/name)
                    if len(paths)>MAX_FILES:raise ValueError('Dataset exceeds 10,000 files')
            paths.sort()
        else:paths=[source]
        items=[];files=[];seen=set();total=0;duplicates=0
        for file in paths:
            suffix=file.suffix.lower()
            if project['task']=='vision' and suffix not in {'.png','.jpg','.jpeg','.bmp','.webp'}:continue
            if project['task']=='llm' and suffix!='.jsonl':continue
            data=bounded_file(file,MAX_BYTES-total);total+=len(data)
            if not data:raise ValueError('Empty dataset file: '+file.name)
            records=[]
            if project['task']=='llm':
                if len(data)>16*1024*1024:raise ValueError('JSONL files must be <=16 MiB')
                for number,line in enumerate(data.decode('utf-8').splitlines(),1):
                    if not line.strip():continue
                    row=json.loads(line)
                    if not isinstance(row,dict):raise ValueError(f'JSONL line {number} must be an object')
                    valid=any(isinstance(row.get(k),str) and bool(row[k].strip()) for k in ['text','content'])
                    valid = valid or (isinstance(row.get('instruction'),str) and bool(row['instruction'].strip()) and isinstance(row.get('output'),str) and bool(row['output'].strip()))
                    msgs=row.get('messages')
                    if isinstance(msgs,list) and msgs:
                        valid=all(isinstance(m,dict) and m.get('role') in ['system','user','assistant'] and isinstance(m.get('content'),str) and bool(m['content'].strip()) for m in msgs)
                    if not valid:raise ValueError(f'JSONL line {number} requires text, content, instruction/output or valid messages')
                    canonical=encode(row).encode();records.append((hashlib.sha256(canonical).hexdigest(),number))
                    if len(items)+len(records)>100000:raise ValueError('Dataset exceeds 100,000 records')
            else:
                png=data.startswith(b'\x89PNG\r\n\x1a\n');jpg=data.startswith(b'\xff\xd8\xff');bmp=data.startswith(b'BM');webp=data.startswith(b'RIFF') and data[8:12]==b'WEBP'
                if not (png or jpg or bmp or webp):raise ValueError('Image signature not recognised: '+file.name)
                records=[(hashlib.sha256(data).hexdigest(),None)]
            digest=self.put_blob(data)
            rel=file.relative_to(source).as_posix() if source.is_dir() else file.name
            files.append({'path':rel,'sha256':digest,'bytes':len(data)})
            for record_hash,line in records:
                if record_hash in seen:duplicates+=1;continue
                seen.add(record_hash);split=split_for(record_hash)
                items.append({'sha256':record_hash,'file_sha256':digest,'path':rel,'line':line,'split':split,'calibration':split=='train'})
        if not items:raise ValueError('No supported data records found')
        counts={key:sum(i['split']==key for i in items) for key in ['train','validation','test']}
        dataset_hash=hashlib.sha256(encode({'files':files,'items':items}).encode()).hexdigest()
        manifest={'id':uid(),'project_id':project_id,'source':str(source),'sha256':dataset_hash,'count':len(items),'duplicates_removed':duplicates,'bytes':total,'split_counts':counts,'license':'unknown','files':files,'items':items,
                  'validation':'JSONL structure or image signature only; labels and content quality not validated',
                  'split_policy':'content-hash 80/10/10; calibration=train; not group-aware','created_at':time.time()}
        with self.connection() as db:db.execute('INSERT INTO datasets VALUES (?,?,?)',(manifest['id'],project_id,encode(manifest)))
        self.record_job('dataset_import',project_id,{'dataset_id':manifest['id'],'count':manifest['count']})
        return manifest
    def plan(self,project_id):
        p=self.project(project_id);datasets=self.datasets(project_id);latest=datasets[-1] if datasets else None
        ready=bool(latest and all(latest['split_counts'].values()))
        stages=[{'name':'Dataset snapshot','status':'ready' if latest else 'blocked','reason':'Snapshot recorded; review labels and leakage.' if latest else 'Import a dataset.'},
                {'name':'Quality gate','status':'blocked','reason':'Review dataset quality and group separation.' if ready else 'Require nonempty train, validation and test splits, then review data.'},
                {'name':'Host training','status':'blocked','reason':'Choose a local or registered Linux worker, compatible model and training environment.' if p['task']=='llm' else 'Configure a supported off-Pi training provider.'},
                {'name':'Hailo compilation','status':'blocked','reason':'Qualified vendor GenAI recipe and SDK required.' if p['task']=='llm' else 'Configure a pinned compatible Linux DFC Python environment or registered worker.'},
                {'name':'Pi deployment/test','status':'blocked','reason':'10H hardware testing deferred; software upload/runtime adapters are available.' if p['target']=='hailo10h' else 'Enroll the Pi and select a compatible bundle to deploy and benchmark.'},
                {'name':'OS image','status':'blocked','reason':'Select an existing OS .img/.img.xz and model bundle; customise a copy using the Linux/Docker backend.'}]
        return {'project':p,'dataset_id':latest['id'] if latest else None,'stages':stages,
                'runtime_package':'hailo-h10-all' if p['target']=='hailo10h' else 'hailo-all',
                'hardware_test':'pending','artifact_kind':'preparation_only'}
    def bundle(self,project_id):
        p=self.project(project_id);datasets=self.datasets(project_id);plan=self.plan(project_id)
        # Verify source snapshots before claiming the preparation artifact is intact.
        for dataset in datasets:
            for f in dataset['files']:
                if hashlib.sha256(bounded_file(self.blobs/f['sha256'])).hexdigest()!=f['sha256']:raise ValueError('Dataset integrity failure')
        path=self.exports/(project_id+'-'+uid()+'.zip')
        payloads={'project.json':encode(p),'plan.json':encode(plan),'datasets.json':encode(datasets),
                  'README.txt':'PREPARATION BUNDLE ONLY. No HEF, model weights, raw training data or bootable image. Dataset manifests refer to local content-addressed blobs. Review docs/architecture.md for the build workflow.',
                  'pi_probe.py':Path(__file__).with_name('pi_probe.py').read_text()}
        payloads['checksums.json']=encode({k:hashlib.sha256(v.encode()).hexdigest() for k,v in payloads.items()})
        with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for name,data in payloads.items():z.writestr(name,data)
        result={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'kind':'preparation_bundle','compiled_model':False,'bootable_image':False}
        self.record_job('preparation_bundle',project_id,result);return result
    def llm_recipe(self,project_id,engine,model,data,max_length=512,rank=8):
        from .llm_tools import prepare_recipe
        p=self.project(project_id)
        if p['task']!='llm':raise ValueError('Choose a Hailo-10H LLM project for a host training recipe')
        recipe_id=uid();output=self.exports/(recipe_id+'-soup.json')
        source=Path(data).expanduser()
        if not source.is_file() or source.suffix.lower()!='.jsonl':raise ValueError('Recipe data must be a single Alpaca JSONL file')
        dataset=self.import_dataset(project_id,data)
        source_lines=bounded_file(self.blobs/dataset['files'][0]['sha256']).decode('utf-8').splitlines()
        train_lines=[source_lines[item['line']-1] for item in dataset['items'] if item['split']=='train']
        training_hash=self.put_blob(('\n'.join(train_lines)+'\n').encode())
        result=prepare_recipe(engine,model,str(self.blobs/training_hash),str(output),str(self.exports/(recipe_id+'-adapters')),max_length,rank)
        result['dataset_id']=dataset['id']
        result['training_records']=len(train_lines)
        result['notes'].append('Only snapshot training records are passed to the recipe; snapshot validation and test records are excluded.')
        self.record_job('llm_recipe',project_id,result);return result
    def score(self,tasks_path,outputs_path):
        from .llm_tools import score_outputs
        return score_outputs(tasks_path,outputs_path)
    def estimate(self,parameters):
        from .llm_tools import estimate
        if not isinstance(parameters,dict):raise ValueError('parameters must be an object')
        return estimate(**parameters)
    def run_job(self,project_id,kind,spec):
        from .jobs import enqueue
        return enqueue(self,project_id,kind,spec)
    def workflow_jobs(self):
        from .jobs import list_all
        return list_all(self)
    def job_get(self,job_id):
        from .jobs import get
        return get(self,job_id)
    def job_cancel(self,job_id):
        from .jobs import cancel
        return cancel(self,job_id)
    def job_logs(self,job_id):
        from .jobs import job_logs
        return job_logs(self,job_id)
    def doctor(self):
        from .workflows import doctor
        result=doctor()
        from .runtime import suggested_python
        result['suggested_training_python']=suggested_python()
        return result
    def annotate(self,project_id,dataset_id,annotations_path):
        from .data import annotate
        return annotate(self,project_id,dataset_id,annotations_path)
    def device_enroll(self,name,host):
        from .workflows import enroll
        return enroll(self,name,host)
    def devices(self):
        from .workflows import settings_get
        return settings_get(self,'devices',[])
    def worker_enroll(self,name,host,python):
        from .remote import register_worker
        return register_worker(self,name,host,python)
    def workers(self):
        from .workflows import settings_get
        return settings_get(self,'workers',[])
    def upload_create(self,project_id):
        from .workflows import settings_set
        self.project(project_id);upload_id=uid();folder=self.root/'uploads'/upload_id;folder.mkdir(parents=True)
        result={'id':upload_id,'project_id':project_id,'bytes':0,'files':0,'status':'open'}
        settings_set(self,'upload:'+upload_id,result);return result
    def upload_file(self,project_id,upload_id,relative_path,data_base64):
        import base64
        from pathlib import PurePosixPath
        from .workflows import settings_get,settings_set
        self.project(project_id);entry=settings_get(self,'upload:'+upload_id,None)
        if not entry or entry['project_id']!=project_id or entry['status']!='open':raise ValueError('Invalid or finalised upload')
        parts=PurePosixPath(relative_path)
        if not parts.parts or parts.is_absolute() or any(x in ('.','..') for x in parts.parts) or '\\' in relative_path or ':' in relative_path or len(parts.parts)>8:raise ValueError('Unsafe upload filename')
        if len(data_base64)>16*1024*1024:raise ValueError('Upload file must be <=12 MiB; use local folder import for larger files')
        data=base64.b64decode(data_base64,validate=True)
        if entry['bytes']+len(data)>MAX_BYTES or entry['files']>=MAX_FILES:raise ValueError('Upload exceeds dataset limits')
        folder=self.root/'uploads'/entry['id'];dest=folder.joinpath(*parts.parts)
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as stream:stream.write(data)
        entry['bytes']+=len(data);entry['files']+=1;settings_set(self,'upload:'+upload_id,entry)
        return {'id':upload_id,'files':entry['files'],'bytes':entry['bytes']}
    def upload_finish(self,project_id,upload_id):
        from .workflows import settings_get,settings_set
        entry=settings_get(self,'upload:'+upload_id,None)
        if not entry or entry['project_id']!=project_id:raise ValueError('Unknown project upload')
        if entry['status']=='imported':return next(d for d in self.datasets(project_id) if d['id']==entry['dataset_id'])
        result=self.import_dataset(project_id,str(self.root/'uploads'/entry['id']))
        entry['status']='imported';entry['dataset_id']=result['id'];settings_set(self,'upload:'+upload_id,entry);return result

    def write_script(self,project_id,name,content):
        import re
        self.project(project_id)
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}',name) or Path(name).suffix not in ['.py','.sh','.json','.yaml','.yml','.txt','.md']:raise ValueError('Choose a simple script/config filename with an allowed extension')
        if not isinstance(content,str) or len(content.encode())>1024*1024:raise ValueError('Script content must be <=1 MiB')
        folder=self.root/'projects'/project_id/'scripts';folder.mkdir(parents=True,exist_ok=True)
        path=folder/name;previous=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        path.write_text(content,encoding='utf-8')
        return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'previous_sha256':previous,'executed':False}
    def project_files(self,project_id):
        self.project(project_id);folder=self.root/'projects'/project_id/'scripts'
        return {'scripts_dir':str(folder),'files':[{'name':p.name,'path':str(p),'bytes':p.stat().st_size} for p in sorted(folder.glob('*')) if p.is_file() and not p.is_symlink()]}
    def prepare_imager(self,image_path):
        from .imager import prepare
        return prepare(self,image_path)
    def open_imager(self,image_path):
        from .imager import open_imager
        return open_imager(self,image_path)
