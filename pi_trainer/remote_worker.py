"""Transferred worker entrypoint; executes a single structured provider operation."""
import hashlib
import json
import platform
import traceback
import zipfile
from pathlib import Path

def run(folder):
    folder=Path(folder)
    if platform.system()!='Linux':raise RuntimeError('Registered compiler/image/training SSH worker must be Linux')
    request=json.loads((folder/'request.json').read_text());spec=request['spec']
    for key,path in request['path_fields'].items():spec[key]=str(folder/path)
    output=folder/'outputs';output.mkdir(exist_ok=False)
    def emit(event):
        with (output/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
    cancelled=lambda:(folder/'cancel').exists()
    try:
        if request['kind']=='train':
            from .training import run_training
            result=run_training(spec,output,emit,cancelled)
        elif request['kind']=='optimise':
            from .optimisation import optimise
            result=optimise(spec,output,emit,cancelled)
        elif request['kind']=='compile':
            from .compiler import run_compile
            result=run_compile(spec,output,emit,cancelled)
        elif request['kind']=='image':
            from .images import customise_image
            result=customise_image(spec,output,emit,cancelled)
        else:raise ValueError('Unsupported worker operation')
        report={'status':'succeeded','result':result}
    except Exception as exc:report={'status':'failed','error':f'{type(exc).__name__}: {exc}'}
    files={}
    for file in output.rglob('*'):
        if file.is_file() and not file.is_symlink():
            with file.open('rb') as stream:files['outputs/'+file.relative_to(output).as_posix()]=hashlib.file_digest(stream,'sha256').hexdigest()
    report['files']=files
    report['output_root']=str(output.resolve())
    with zipfile.ZipFile(folder/'output.zip','x',zipfile.ZIP_STORED,allowZip64=True) as z:
        z.writestr('result.json',json.dumps(report,allow_nan=False))
        for name in files:z.write(folder/name,name)
    print(json.dumps({'status':report['status'],'artifact_files':len(files)}))
