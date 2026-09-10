"""Real vendor-SDK smoke test; synthetic vision fixture, no hardware/LLM claim.

Run inside the licensed SDK environment:
  python hailo_sdk_smoke.py --target hailo8l --output /work/run-8l \
    --compiler-source /workspace/pi_trainer/compiler.py

Each stage is bounded. Outputs must be new. Numerical emulation uses the vendor
documented ClientRunner.infer API, inspected in SDK 5.4.0; no HEF emulator is invented.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback

SOURCES = [
    'https://github.com/hailo-ai/hailo_model_zoo/blob/master/hailo_model_zoo/core/infer/model_infer.py',
    'https://github.com/hailo-ai/hailo_model_zoo/blob/master/hailo_model_zoo/main_driver.py',
    'https://github.com/hailo-ai/hailo_model_zoo/blob/master/hailo_model_zoo/cfg/alls/generic/swin_small.alls',
]

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def progress(stage,**details):
    print(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(),stage=stage,**details)),flush=True)

def execute(argv,log,timeout,monitor=()):
    with Path(log).open('wb') as stream:
        process=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        started=time.monotonic();observed=set()
        while process.poll() is None:
            for stage,path in monitor:
                if stage not in observed and path.is_file():
                    observed.add(stage);progress(stage,artifact=str(path),elapsed_seconds=time.monotonic()-started)
            if time.monotonic()-started>timeout:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise TimeoutError('Stage exceeded %ss; see %s'%(timeout,log))
            time.sleep(0.5)
        code=process.returncode
    if code:raise RuntimeError('Stage exited %s; see %s'%(code,log))

def emulate(har,mode,samples_path,result_path):
    import numpy as np
    from hailo_sdk_client import ClientRunner,InferenceContext
    runner=ClientRunner(har=str(har))
    selected=getattr(InferenceContext,mode)
    samples=np.load(samples_path,allow_pickle=False)
    # Inspected SDK 5.4.0 ClientRunner.infer docstring accepts an NHWC NumPy
    # array and returns one entry per INPUT sample (not per output layer).
    # It converts the array to the dataset structure required by model.build.
    with runner.infer_context(selected) as context:
        value=runner.infer(context,samples,batch_size=1)
    array=np.asarray(value)
    if array.shape[0]!=len(samples) or array.size!=len(samples)*8 or not np.isfinite(array).all():
        raise ValueError('Unexpected or non-finite fixture output: %s'%(array.shape,))
    np.save(Path(result_path).with_suffix('.npy'),array)
    Path(result_path).write_text(json.dumps({'status':'passed','mode':mode,'api':'ClientRunner.infer(context, NHWC_numpy, batch_size=1)','shape':list(array.shape),'finite':True,'minimum':float(array.min()),'maximum':float(array.max()),'output_sha256':sha256(Path(result_path).with_suffix('.npy'))},indent=2))

def run(args,output):
    import numpy as np
    import onnx
    from onnx import helper,numpy_helper,TensorProto
    report={'schema':'pi-trainer/vendor-sdk-smoke/v1','target':args.target,'platform':platform.platform(),'python':platform.python_version(),'sources':SOURCES,'fixture_only':True,'hardware_tested':False,'llm_qualified':False,'interpretation':'Real SDK parsing, quantization, compilation and numerical emulation of a synthetic vision fixture only; not task accuracy, HEF hardware execution, NPU speed or LLM compatibility.'}
    report['versions']={}
    for name in ('hailo-dataflow-compiler','hailo-sdk-client','numpy','onnx','tensorflow'):
        try:report['versions'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    import hailo_sdk_client
    report['sdk_module_version']=str(getattr(hailo_sdk_client,'__version__','not exposed'))
    rng=np.random.default_rng(4242)
    weights=rng.uniform(-0.2,0.3,(8,3,1,1)).astype(np.float32)
    bias=np.full((8,),0.1,dtype=np.float32)
    graph=helper.make_graph([
        helper.make_node('Conv',['images','weights','bias'],['conv'],name='conv',kernel_shape=[1,1]),
        helper.make_node('Relu',['conv'],['relu'],name='relu'),
        helper.make_node('GlobalAveragePool',['relu'],['scores'],name='pool')],
        'sdk_fixture',[helper.make_tensor_value_info('images',TensorProto.FLOAT,[1,3,32,32])],
        [helper.make_tensor_value_info('scores',TensorProto.FLOAT,[1,8,1,1])],
        initializer=[numpy_helper.from_array(weights,'weights'),numpy_helper.from_array(bias,'bias')])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)],producer_name='pi-trainer-synthetic-sdk-smoke')
    model.ir_version=8;onnx.checker.check_model(model);onnx.save(model,output/'fixture.onnx')
    calibration=rng.uniform(0,1,(64,32,32,3)).astype(np.float32)
    samples=rng.uniform(0,1,(2,32,32,3)).astype(np.float32)
    np.save(output/'calibration.npy',calibration);np.save(output/'samples.npy',samples)
    expected=np.maximum(np.einsum('nhwc,oc->nhwo',samples,weights[:,:,0,0])+bias,0).mean(axis=(1,2))
    np.save(output/'numpy-reference.npy',expected)
    script=output/'fixture.alls'
    script.write_text('model_optimization_flavor(optimization_level=0, compression_level=0)\n')
    report['fixture']={'onnx_sha256':sha256(output/'fixture.onnx'),'calibration_sha256':sha256(output/'calibration.npy'),'calibration_samples':64,'input_layout':'NCHW ONNX, NHWC SDK','output_channels':8,'optimization_level':0,'compression_level':0,'production_recommendation':False}
    build=output/'compiled';build.mkdir()
    config={'target':args.target,'task':'vision','model_path':str(output/'fixture.onnx'),'calibration_path':str(output/'calibration.npy'),'output_dir':str(build),'model_script_path':str(script)}
    config_path=output/'compile-config.json';config_path.write_text(json.dumps(config,indent=2))
    started=time.monotonic();progress('compile_worker_start',target=args.target,log=str(output/'compile.log'))
    try:
        execute([sys.executable,str(Path(args.compiler_source).resolve()),str(config_path)],output/'compile.log',args.timeout,monitor=[('parsed_har_ready',build/'parsed.har'),('optimized_har_ready',build/'optimized.har'),('hef_ready',build/'model.hef')])
        hef=build/'model.hef'
        if not hef.is_file() or not hef.stat().st_size:raise ValueError('No real HEF produced')
        manifest=json.loads((build/'compile-result.json').read_text())
        if manifest['target']!=args.target:raise ValueError('Wrong compiler target')
        report['compilation']={'status':'passed','bytes':hef.stat().st_size,'sha256':sha256(hef),'elapsed_seconds':time.monotonic()-started,'application_worker_sha256':sha256(args.compiler_source)}
    except Exception as exc:report['compilation']={'status':'failed','error':str(exc)}
    progress('compile_worker_end',**report['compilation'])
    for mode,har in [('SDK_NATIVE',build/'parsed.har'),('SDK_QUANTIZED',build/'optimized.har')]:
        if not har.is_file():
            report[mode]={'status':'blocked','reason':'Required HAR was not produced'};continue
        result_path=output/(mode.lower()+'.json');progress('emulation_start',mode=mode)
        try:
            execute([sys.executable,str(Path(__file__).resolve()),'--emulate-mode',mode,'--har',str(har),'--samples',str(output/'samples.npy'),'--emulation-result',str(result_path)],output/(mode.lower()+'.log'),args.timeout)
            result=json.loads(result_path.read_text());actual=np.load(result_path.with_suffix('.npy'),allow_pickle=False).reshape(2,8)
            result['numpy_reference_max_absolute_error']=float(np.max(np.abs(actual-expected)))
            if mode=='SDK_NATIVE' and not np.allclose(actual,expected,atol=1e-4,rtol=1e-4):raise ValueError('Native SDK output disagrees with independent NumPy reference')
            result['quality_claim']=False;report[mode]=result
        except Exception as exc:report[mode]={'status':'failed','error':str(exc)}
        progress('emulation_end',mode=mode,status=report[mode]['status'])
    report['status']='passed' if all(report[key]['status']=='passed' for key in ('compilation','SDK_NATIVE','SDK_QUANTIZED')) else 'incomplete'
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',choices=['hailo8l','hailo8','hailo10h']);parser.add_argument('--output')
    parser.add_argument('--compiler-source',default=str(Path(__file__).resolve().parents[1]/'pi_trainer/compiler.py'))
    parser.add_argument('--timeout',type=int,default=1200)
    parser.add_argument('--emulate-mode',choices=['SDK_NATIVE','SDK_QUANTIZED']);parser.add_argument('--har');parser.add_argument('--samples');parser.add_argument('--emulation-result')
    args=parser.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1');os.environ.setdefault('OMP_NUM_THREADS','2')
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
    if args.emulate_mode:
        emulate(args.har,args.emulate_mode,args.samples,args.emulation_result);return 0
    if not args.target or not args.output:parser.error('--target and --output are required')
    if not 1<=args.timeout<=7200:parser.error('--timeout must be1..7200 seconds per stage')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):parser.error('Output must be empty to preserve evidence')
    progress('sdk_smoke_start',target=args.target,output=str(output))
    try:report=run(args,output)
    except Exception as exc:report={'status':'failed','target':args.target,'error':str(exc),'traceback':traceback.format_exc(),'hardware_tested':False,'llm_qualified':False}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))
    return 0 if report['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())
