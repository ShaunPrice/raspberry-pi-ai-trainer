"""Hailo DFC compilation and explicit vendor GenAI recipe boundary."""
from __future__ import annotations
import json
from pathlib import Path
import sys
if __package__:
    from .training import artifact, TARGETS
else:
    TARGETS = {"hailo8l", "hailo8", "hailo10h"}

def run_compile(spec:dict,workdir:Path,emit,cancelled)->dict:
    from .processes import run_process
    if spec.get('target') not in TARGETS:raise ValueError('Unknown Hailo target')
    if spec.get('task','vision') not in ('vision','llm'):raise ValueError('Unknown task')
    workdir=Path(workdir).resolve();workdir.mkdir(parents=True,exist_ok=True)
    model_value=spec.get('model_path') or spec.get('model')
    if not model_value:raise ValueError('Compilation model path is required')
    model=Path(model_value)
    if not model.exists():raise ValueError('Compilation input does not exist')
    config=dict(spec,model_path=str(model.resolve()),output_dir=str(workdir))
    if spec.get('model_script_path'):
        script=Path(spec['model_script_path']).resolve()
        if not script.is_file():raise ValueError('Model script does not exist')
        config['model_script_path']=str(script)
    config_path=workdir/'compile-config.json';config_path.write_text(json.dumps(config,indent=2))
    if spec.get('task')=='llm':
        if spec['target']!='hailo10h':raise ValueError('GenAI compilation requires Hailo-10H')
        # This path is trusted operator configuration, never a shell string.
        recipe=spec.get('recipe_executable')
        if not recipe or not Path(recipe).is_absolute() or not Path(recipe).is_file():
            raise ValueError('Hailo-10H GenAI requires an installed vendor-qualified recipe_executable (absolute path); arbitrary GGUF/LoRA to HEF conversion is not supported')
        argv=[str(recipe),'--pi-trainer-config',str(config_path)]
    else:
        calibration=Path(spec.get('calibration_path') or '')
        if model.suffix.lower()!='.onnx' or not model.is_file():raise ValueError('Vision compiler requires an ONNX model')
        if not calibration.is_file() or calibration.suffix!='.npy':raise ValueError('Provide calibration_path: representative float32 NHWC .npy data matching model preprocessing')
        config['calibration_path']=str(calibration.resolve());config_path.write_text(json.dumps(config,indent=2))
        from .runtime import compute_python
        python=compute_python(spec)
        argv=[python,str(Path(__file__).resolve()),str(config_path)]
    emit({'stage':'compiling','message':'Launching target-specific Hailo compiler'})
    run_process(argv,cwd=workdir,log_path=workdir/'compile.log',cancelled=cancelled,timeout=float(spec.get('timeout',86400)))
    manifest=workdir/'compile-result.json'
    if not manifest.is_file():raise ValueError('Compiler did not write compile-result.json')
    result=json.loads(manifest.read_text())
    if result.get('target')!=spec['target']:raise ValueError('Compiler returned the wrong target')
    paths=result.pop('artifact_paths',[])
    if not paths:raise ValueError('Compiler returned no artifacts')
    verified=[]
    for raw in paths:
        path=(workdir/str(raw)).resolve()
        if not path.is_relative_to(workdir):raise ValueError('Compiler artifact escaped its job directory')
        verified.append(artifact(path))
    if not any(Path(a['path']).suffix=='.hef' for a in verified):raise ValueError('Compiler did not produce a HEF')
    result['model_path']=next(a['path'] for a in verified if Path(a['path']).suffix=='.hef')
    result.update(artifacts=verified,evidence='compiler output exists and is hashed; physical target load and inference unverified')
    (workdir/'result.json').write_text(json.dumps(result,indent=2));emit({'stage':'completed','artifacts':verified})
    return result

def _worker(config):
    try:
        import numpy as np
        from hailo_sdk_client import ClientRunner
    except ImportError as exc:raise RuntimeError('Install the licensed Hailo Dataflow Compiler for this target in a supported Linux x86_64 environment, then select its Python interpreter') from exc
    calibration=np.load(config['calibration_path'],allow_pickle=False,mmap_mode='r')
    if calibration.ndim!=4 or calibration.shape[0]<1 or calibration.shape[-1] not in (1,3,4) or calibration.dtype!=np.float32:
        raise ValueError('Calibration must be nonempty float32 NHWC image data')
    if not np.isfinite(calibration).all():raise ValueError('Calibration contains non-finite values')
    runner=ClientRunner(hw_arch=config['target'])
    kwargs={}
    for key in ('start_node_names','end_node_names','net_input_shapes'):
        if key in config:kwargs[key]=config[key]
    runner.translate_onnx_model(config['model_path'],'pi_trainer_model',**kwargs)
    if config.get('model_script_path'):runner.load_model_script(Path(config['model_script_path']).read_text())
    out=Path(config['output_dir']);runner.save_har(str(out/'parsed.har'))
    runner.optimize(calibration);runner.save_har(str(out/'optimized.har'))
    hef=runner.compile()
    if not isinstance(hef,(bytes,bytearray)) or not hef:raise ValueError('DFC returned no compiled HEF bytes')
    (out/'model.hef').write_bytes(hef)
    (out/'compile-result.json').write_text(json.dumps({'target':config['target'],'provider':'hailo-dfc','artifact_paths':['model.hef','parsed.har','optimized.har'],'calibration_samples':int(calibration.shape[0])}))

if __name__=='__main__':
    # Support direct execution from any selected SDK interpreter without installing core.
    _worker(json.loads(Path(sys.argv[1]).read_text()))
