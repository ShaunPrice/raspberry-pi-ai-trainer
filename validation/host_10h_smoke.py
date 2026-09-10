"""Portable, offline 10H-targeted host validation; does not emulate the Hailo NPU.

Creates synthetic fixtures, exercises real training/export/adapter reload, and
reports unavailable vendor/hardware gates explicitly. No pretrained downloads.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def run(output, device):
    import numpy as np
    from PIL import Image
    import torch
    from torch import nn
    from onnx.reference import ReferenceEvaluator
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, AutoModelForCausalLM, AutoTokenizer
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from peft import PeftModel
    from pi_trainer.training import run_training
    torch.set_num_threads(2)
    torch.manual_seed(20260910)
    start = time.monotonic()
    report = {'schema':'pi-trainer/host-10h-validation/v1', 'target':'hailo10h',
              'platform':platform.system(), 'architecture':platform.machine(),
              'python':platform.python_version(), 'device':device,
              'model_origin':'locally generated random tiny GPT2 and synthetic colour classifier',
              'deployment_model_tested':False, 'hardware_tested':False,
              'versions':{name:importlib.metadata.version(name) for name in ['torch','numpy','pillow','onnx','transformers','peft']}}
    report['accelerators']={'cuda':torch.cuda.is_available(), 'mps':bool(hasattr(torch.backends,'mps') and torch.backends.mps.is_available())}
    if device == 'cuda' and not report['accelerators']['cuda']:raise RuntimeError('CUDA unavailable')
    if device == 'mps' and not report['accelerators']['mps']:raise RuntimeError('MPS unavailable')
    images=output/'vision-data'
    for subset,count in [('train',12),('validation',4),('test',4)]:
        for channel,label in [(0,'red'),(2,'blue')]:
            folder=images/subset/label;folder.mkdir(parents=True)
            for index in range(count):
                seed={'train':0,'validation':100,'test':200}[subset]+index+channel*1000
                rng=np.random.default_rng(seed)
                pixels=rng.integers(0,20,size=(16,16,3),dtype=np.uint8);pixels[:,:,channel]+=180
                Image.fromarray(pixels).save(folder/f'{index}.png')
    events=[]
    vision=run_training({'target':'hailo10h','task':'vision','dataset_dir':str(images),
                        'python':sys.executable,'device':device,'image_size':16,'epochs':2,'seed':20260910,'threads':2},output/'vision',events.append,lambda:False)
    checkpoint=torch.load(output/'vision/model.pt',map_location='cpu',weights_only=True)
    model=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(32,2))
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    x=np.random.default_rng(7).random((1,3,16,16),dtype=np.float32)
    with torch.no_grad():reference=model(torch.from_numpy(x)).numpy()
    exported=ReferenceEvaluator(str(output/'vision/model.onnx')).run(None,{'images':x})[0]
    np.testing.assert_allclose(exported,reference,rtol=1e-4,atol=1e-5)
    calibration=np.load(output/'vision/calibration.npy',allow_pickle=False)
    assert calibration.shape==(24,16,16,3) and calibration.dtype==np.float32
    assert np.isfinite(calibration).all() and calibration.min()>=0 and calibration.max()<=1
    split=json.loads((output/'vision/dataset-split.json').read_text())
    hashes={s:{r['sha256'] for r in split if r['split']==s} for s in ['train','validation','test']}
    assert not (hashes['train'] & hashes['test'] or hashes['train'] & hashes['validation'] or hashes['validation'] & hashes['test'])
    report['vision']={'status':'passed','metrics':vision['metrics'],'onnx_max_absolute_error':float(np.max(np.abs(exported-reference))),
                      'calibration_train_only':True,'disjoint_splits':True,'onnx_sha256':digest(output/'vision/model.onnx')}
    base=output/'llm-base';base.mkdir()
    tokens=['[UNK]','[PAD]','[EOS]','###','Instruction','Input','Response',':','Task','answer','value']+[str(i) for i in range(40)]
    tokenizer=Tokenizer(WordLevel({word:i for i,word in enumerate(tokens)},unk_token='[UNK]'));tokenizer.pre_tokenizer=Whitespace()
    fast=PreTrainedTokenizerFast(tokenizer_object=tokenizer,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]')
    fast.save_pretrained(base)
    torch.manual_seed(20260910)
    GPT2LMHeadModel(GPT2Config(vocab_size=len(tokens),n_positions=128,n_ctx=128,n_embd=32,n_layer=1,n_head=2,
                             bos_token_id=2,eos_token_id=2,pad_token_id=1)).save_pretrained(base,safe_serialization=True)
    data=output/'llm-data';data.mkdir()
    for subset,indices in [('train',range(12)),('validation',range(12,16)),('test',range(16,20))]:
        (data/(subset+'.jsonl')).write_text('\n'.join(json.dumps({'instruction':f'Task {i}','input':f'value {i}','output':f'answer {i}'}) for i in indices)+'\n')
    llm=run_training({'target':'hailo10h','task':'llm','dataset_dir':str(data),'model':str(base),'engine':'transformers-peft',
                     'python':sys.executable,'device':device,'rank':2,'epochs':1,'max_length':64,'batch_size':2,'threads':2,'allow_download':False},output/'llm',events.append,lambda:False)
    loaded_tokenizer=AutoTokenizer.from_pretrained(output/'llm/adapter',local_files_only=True,trust_remote_code=False)
    inputs=loaded_tokenizer('Task 21 value 21',return_tensors='pt')
    model_base=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False).eval()
    with torch.no_grad():before=model_base(**inputs).logits.detach().clone()
    reloaded=PeftModel.from_pretrained(model_base,output/'llm/adapter',local_files_only=True).eval()
    with torch.no_grad():after=reloaded(**inputs).logits.detach().clone()
    assert torch.isfinite(after).all()
    delta=float((after-before).abs().max());assert delta>0,'Training did not change model outputs'
    merged=reloaded.merge_and_unload().eval()
    with torch.no_grad():merged_logits=merged(**inputs).logits.detach().clone()
    torch.testing.assert_close(merged_logits,after,rtol=1e-4,atol=1e-5)
    generated=merged.generate(**inputs,max_new_tokens=8,do_sample=False,pad_token_id=1,eos_token_id=2)
    new_tokens=int(generated.shape[1]-inputs['input_ids'].shape[1]);assert 1<=new_tokens<=8
    assert math.isfinite(llm['metrics']['validation_loss']) and math.isfinite(llm['metrics']['test_perplexity'])
    report['llm']={'status':'passed','metrics':llm['metrics'],'adapter_reload':True,'adapter_changed_logits_max':delta,
                   'adapter_merge_max_absolute_error':float((merged_logits-after).abs().max()),'generated_tokens':new_tokens,
                   'adapter_sha256':digest(output/'llm/adapter/adapter_model.safetensors')}
    sdk=importlib.util.find_spec('hailo_sdk_client') is not None
    report['hailo_sdk_installed']=sdk
    report['hailo_emulation']={'status':'blocked','reason':'Compatible Hailo DFC and model-specific numerical-emulation recipe have not been configured.' if sdk else 'Hailo Dataflow Compiler is not installed in this test environment.'}
    report['hailo_compilation']={'status':'blocked','reason':'Requires the qualified target SDK and a vendor-supported model recipe; synthetic fixture is not a qualified Hailo LLM.'}
    report['hef_execution']={'status':'not_run','reason':'No Hailo-10H hardware is attached.'}
    report['status']='passed';report['elapsed_seconds']=time.monotonic()-start
    report['interpretation']='Passed means host pipeline only. No useful language-quality, Hailo numerical-emulation, compiled HEF, NPU performance or final model-fit claim.'
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu');args=parser.parse_args()
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise SystemExit('Output directory must be empty to preserve previous evidence')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    try:report=run(output,args.device)
    except Exception as exc:
        report={'status':'failed','platform':platform.system(),'device':args.device,'error':str(exc),'traceback':traceback.format_exc()}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))
    return 0 if report['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
