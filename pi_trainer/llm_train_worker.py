"""Offline-by-default Transformers/PEFT causal language-model LoRA training."""
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

def main(config):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:raise RuntimeError('LLM worker requires torch, transformers, peft and safetensors in the selected Python environment') from exc
    torch.set_num_threads(int(config.get('threads',2)))
    model_path=config.get('model')
    if not model_path:raise ValueError('A local Hugging Face model directory is required')
    offline=not config.get('allow_download',False)
    if offline and not Path(model_path).is_dir():raise ValueError('Offline model must be an existing local directory; no model downloads are implicit')
    data=Path(config['dataset_path'])
    explicit=data.is_dir() and (data/'train.jsonl').is_file() and (data/'test.jsonl').is_file()
    explicit_root=data if explicit else None
    if explicit:data=data/'train.jsonl'
    if data.is_dir():
        candidates=list(data.rglob('*.jsonl'))
        if len(candidates)!=1:raise ValueError('Select exactly one Alpaca JSONL dataset')
        data=candidates[0]
    if data.stat().st_size>256*1024*1024:raise ValueError('JSONL input limit is 256 MiB')
    rows=[json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    if len(rows)<10:raise ValueError('At least 10 Alpaca records are required for held-out evaluation')
    train_rows=len(rows)
    if explicit:
        rows += [json.loads(line) for line in (explicit_root/'test.jsonl').read_text().splitlines() if line.strip()]
        if len(rows)==train_rows:raise ValueError('Explicit test set is empty')
    test_end=len(rows)
    if explicit and (explicit_root/'validation.jsonl').is_file():
        rows += [json.loads(line) for line in (explicit_root/'validation.jsonl').read_text().splitlines() if line.strip()]
    texts=[];seen=set()
    for row in rows:
        if not isinstance(row,dict) or not isinstance(row.get('instruction'),str) or not isinstance(row.get('output'),str) or not row['instruction'].strip() or not row['output'].strip():raise ValueError('Each record needs nonempty instruction and output strings')
        if not isinstance(row.get('input',''),str):raise ValueError('input must be a string')
        text=f"### Instruction:\n{row['instruction']}\n### Input:\n{row.get('input','')}\n### Response:\n{row['output']}"
        digest=hashlib.sha256(text.encode()).hexdigest()
        if digest in seen:raise ValueError('Duplicate training records detected; remove duplicates before splitting')
        seen.add(digest);texts.append(text)
    seed=int(config.get('seed',42));random.seed(seed);torch.manual_seed(seed)
    if explicit:train=texts[:train_rows];test=texts[train_rows:test_end];validation=texts[test_end:]
    else:
        random.shuffle(texts);holdout=max(1,len(texts)//5);test=texts[:holdout];validation=texts[holdout:2*holdout];train=texts[2*holdout:]
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=offline,trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:raise ValueError('Tokenizer needs EOS or padding token')
        tokenizer.pad_token=tokenizer.eos_token
    base=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=offline,trust_remote_code=False)
    model=get_peft_model(base,LoraConfig(task_type='CAUSAL_LM',r=int(config.get('rank',8)),lora_alpha=int(config.get('rank',8))*2,lora_dropout=0.05,target_modules=config.get('target_modules','all-linear')))
    device=config.get('device','cpu')
    if device not in ('cpu','cuda','mps'):raise ValueError('device must be cpu, cuda or mps')
    model.to(device);optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=float(config.get('learning_rate',0.0002)))
    batch=int(config.get('batch_size',1));max_length=int(config.get('max_length',256))
    def batches(records):
        for i in range(0,len(records),batch):
            encoded=tokenizer(records[i:i+batch],padding=True,truncation=True,max_length=max_length,return_tensors='pt')
            encoded={k:v.to(device) for k,v in encoded.items()};labels=encoded['input_ids'].clone();labels[encoded['attention_mask']==0]=-100
            if int((labels[:,1:]!=-100).sum())==0:raise ValueError('No supervised tokens after tokenization')
            yield dict(encoded,labels=labels)
    history=[];start=time.monotonic()
    for epoch in range(int(config.get('epochs',1))):
        model.train();random.shuffle(train);total=0;count=0
        for inputs in batches(train):
            optimizer.zero_grad();loss=model(**inputs).loss
            if not torch.isfinite(loss):raise ValueError('Training produced a non-finite loss')
            loss.backward();optimizer.step();total+=loss.item();count+=1
        history.append({'epoch':epoch+1,'train_loss':total/count});print(json.dumps(history[-1]),flush=True)
    model.eval();loss_sum=0;tokens=0
    with torch.no_grad():
        for inputs in batches(test):
            n=int((inputs['labels'][:,1:]!=-100).sum());loss_sum+=model(**inputs).loss.item()*n;tokens+=n
    validation_metrics={}
    if validation:
        val_sum=0;val_tokens=0
        with torch.no_grad():
            for inputs in batches(validation):
                n=int((inputs['labels'][:,1:]!=-100).sum());val_sum+=model(**inputs).loss.item()*n;val_tokens+=n
        validation_metrics={'validation_loss':val_sum/val_tokens,'validation_records':len(validation)}
    loss=loss_sum/tokens;out=Path(config['output_dir']);adapter=out/'adapter'
    model.save_pretrained(adapter,safe_serialization=True);tokenizer.save_pretrained(adapter)
    if not (adapter/'adapter_model.safetensors').is_file():raise ValueError('PEFT failed to produce a safetensors adapter')
    (out/'dataset-split.json').write_text(json.dumps({'train':[hashlib.sha256(x.encode()).hexdigest() for x in train],'test':[hashlib.sha256(x.encode()).hexdigest() for x in test],'validation':[hashlib.sha256(x.encode()).hexdigest() for x in validation]},indent=2))
    result={'artifact_paths':[str(p) for p in adapter.rglob('*') if p.is_file()]+[str(out/'dataset-split.json')],
            'metrics':{**validation_metrics,'test_loss':loss,'test_perplexity':math.exp(min(loss,80)),'perplexity_capped':loss>80,'test_records':len(test),'train_records':len(train),'test_tokens':tokens,'training_seconds':time.monotonic()-start},'history':history,
            'limitations':['LoRA host checkpoint only; compile a vendor-supported base/adapter pair for Hailo-10H.','Loss includes prompt and response tokens. Task-quality evaluation and semantic/group leakage review remain separate requirements.']}
    (out/'training-result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
if __name__=='__main__':main(json.loads(Path(sys.argv[1]).read_text()))
