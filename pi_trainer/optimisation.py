"""Bounded real training sweeps with quality-first Pareto selection."""
from __future__ import annotations
import itertools
import json
import math
from pathlib import Path
from .processes import JobCancelled

def optimise(spec,workdir,emit,cancelled):
    from .training import run_training
    choices=spec.get('search',{'rank':[4,8],'epochs':[1]} if spec.get('task')=='llm' else {'image_size':[32,64],'epochs':[1]})
    if not isinstance(choices,dict) or set(choices)-{'image_size','epochs','learning_rate','rank','batch_size'}:raise ValueError('Unsupported optimisation search dimensions')
    if any(not isinstance(v,list) or not v for v in choices.values()):raise ValueError('Search dimensions must be nonempty lists')
    count=math.prod(len(v) for v in choices.values())
    max_trials=spec.get('max_trials',4)
    if type(max_trials) is not int or not 1<=max_trials<=12 or count>max_trials:raise ValueError('Search exceeds bounded trial count (maximum 12)')
    minimum=spec.get('min_quality',0.0)
    if type(minimum) not in (int,float) or not 0<=minimum<=1:raise ValueError('min_quality must be 0–1')
    results=[]
    for index,values in enumerate(itertools.product(*choices.values())):
        if cancelled():raise JobCancelled('Optimisation cancelled')
        params=dict(zip(choices,values));trialdir=workdir/f'trial-{index:02d}';trialdir.mkdir()
        emit({'stage':'optimisation','trial':index+1,'total':count,'parameters':params})
        try:
            result=run_training({**spec,**params},trialdir,emit,cancelled)
            # Model selection must use validation; final test remains a separate
            # evidence item, never the ranking criterion.
            metrics=result.get('metrics',{})
            quality=metrics.get('validation_accuracy',metrics.get('val_accuracy'))
            if spec.get('task')=='llm':
                loss=metrics.get('validation_loss')
                quality=math.exp(-loss) if type(loss) in (int,float) and math.isfinite(loss) and loss>=0 else None
            files=[p for p in trialdir.rglob('*') if p.is_file() and p.suffix in ['.onnx','.pt','.safetensors']]
            size=sum(p.stat().st_size for p in files)
            results.append({'index':index,'parameters':params,'status':'succeeded','quality':quality,'artifact_bytes':size,'eligible':quality is not None and quality>=minimum,'result':result})
        except JobCancelled:raise
        except Exception as exc:results.append({'index':index,'parameters':params,'status':'failed','error':str(exc),'eligible':False})
    eligible=[r for r in results if r['eligible']]
    frontier=[r['index'] for r in eligible if not any(o['quality']>=r['quality'] and o['artifact_bytes']<=r['artifact_bytes'] and (o['quality']>r['quality'] or o['artifact_bytes']<r['artifact_bytes']) for o in eligible)]
    result={'trials':results,'pareto_trial_indices':frontier,'metric':('inverse validation perplexity and artifact bytes' if spec.get('task')=='llm' else 'host validation accuracy and artifact bytes'),'evidence':'host training only; Hailo latency/quality requires real device benchmarks'}
    (workdir/'optimisation.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    if not eligible:raise RuntimeError('No trial passed the validation quality gate. Inspect optimisation.json; no winner selected.')
    return result
