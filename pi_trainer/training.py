"""Isolated, cancellable host training providers. Frameworks are optional."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import sys

TARGETS = {'hailo8l', 'hailo8', 'hailo10h'}

def artifact(path):
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'Missing or empty artifact: {path}')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return {'path': str(path), 'sha256': h.hexdigest(), 'bytes': path.stat().st_size}

def validate(spec):
    if spec.get('target') not in TARGETS: raise ValueError('Unknown Hailo target')
    if spec.get('task') not in ('vision', 'llm'): raise ValueError('task must be vision or llm')
    if spec['task'] == 'llm' and spec['target'] != 'hailo10h':
        raise ValueError('LLM deployment requires Hailo-10H; 8/8L are vision targets')
    for name, default, low, high in [('epochs',1,1,1000),('batch_size',8,1,1024),('image_size',64,16,1024),('max_length',256,16,32768),('rank',8,1,256)]:
        value = spec.get(name, default)
        if isinstance(value,bool) or not isinstance(value,int) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    for name, default in [('learning_rate',0.001),('timeout',86400)]:
        value=spec.get(name,default)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:raise ValueError(f'{name} must be finite and positive')
    for p in ('/proc/device-tree/model', '/sys/firmware/devicetree/base/model'):
        try:
            if 'raspberry pi' in Path(p).read_text().lower(): raise ValueError('Training on Raspberry Pi is disabled; use a host workstation')
        except (FileNotFoundError, PermissionError): pass

def run_training(spec:dict, workdir:Path, emit, cancelled)->dict:
    from .processes import run_process
    validate(spec)
    workdir = Path(workdir).resolve(); workdir.mkdir(parents=True,exist_ok=True)
    source_value=spec.get('dataset_dir') or spec.get('dataset_path') or spec.get('path')
    if not source_value:raise ValueError('Training dataset path is required')
    source = Path(source_value)
    if not source.exists(): raise ValueError('Training dataset does not exist')
    config = dict(spec, dataset_path=str(source.resolve()), output_dir=str(workdir))
    if spec['task']=='llm' and spec.get('model') and Path(spec['model']).is_dir():
        config['model']=str(Path(spec['model']).resolve())
    engine = spec.get('engine') or ('pytorch' if spec['task']=='vision' else 'transformers-peft')
    if engine not in ('pytorch','transformers-peft'):
        raise ValueError('Executable engines: pytorch (vision), transformers-peft (LLM). Soup recipes remain available through llm_recipe.')
    if (engine=='pytorch') != (spec['task']=='vision'): raise ValueError('Engine does not match task')
    config_path=workdir/'training-config.json';config_path.write_text(json.dumps(config,indent=2))
    worker=Path(__file__).with_name('vision_train_worker.py' if spec['task']=='vision' else 'llm_train_worker.py')
    emit({'stage':'training','message':f'Launching {engine} in isolated host process'})
    from .runtime import compute_python
    python=compute_python(spec)
    run_process([python,str(worker),str(config_path)],cwd=workdir,
                log_path=workdir/'training.log',cancelled=cancelled,timeout=float(spec.get('timeout',86400)),
                env={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'} if not spec.get('allow_download',False) else None)
    result=json.loads((workdir/'training-result.json').read_text())
    result['artifacts']=[artifact(p) for p in result.pop('artifact_paths')]
    by_name={Path(a['path']).name:a['path'] for a in result['artifacts']}
    if 'model.onnx' in by_name:result['model_path']=by_name['model.onnx']
    if 'calibration.npy' in by_name:result['calibration_path']=by_name['calibration.npy']
    if 'preprocessing.json' in by_name:result['preprocessing_path']=by_name['preprocessing.json']
    if 'adapter_model.safetensors' in by_name:result['adapter_path']=str(Path(by_name['adapter_model.safetensors']).parent)
    result.update(target=spec['target'],task=spec['task'],engine=engine,
                  evidence='host training and held-out evaluation completed; hardware inference unverified')
    (workdir/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    emit({'stage':'completed','metrics':result['metrics']})
    return result
