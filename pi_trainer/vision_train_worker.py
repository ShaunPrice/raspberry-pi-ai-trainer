"""Standalone PyTorch image-folder classifier trainer with ONNX export."""
import hashlib
import json
from pathlib import Path
import random
import sys
import time

IMAGE_EXTENSIONS = ('.jpg','.jpeg','.png','.bmp','.webp')

def validate_explicit_splits(root):
    """Require the same nonempty class coverage before any labels are assigned."""
    root=Path(root)
    subsets=('train','validation','test')
    if not all((root/subset).is_dir() for subset in subsets):
        raise ValueError('Explicit vision splits require train, validation and test directories')
    class_sets={subset:{p.name for p in (root/subset).iterdir() if p.is_dir()} for subset in subsets}
    expected=class_sets['train']
    if any(classes!=expected for classes in class_sets.values()):
        detail='; '.join(f'{subset}={sorted(classes)}' for subset,classes in class_sets.items())
        raise ValueError('Explicit vision splits must contain identical class sets: '+detail)
    if len(expected)<2:raise ValueError('Classification requires at least two classes in every split')
    for subset in subsets:
        for name in expected:
            if not any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in (root/subset/name).rglob('*')):
                raise ValueError(f'Explicit vision split {subset} has no supported images for class {name}')
    return sorted(expected)

def main(config):
    try:
        import numpy as np
        from PIL import Image
        import torch
        from torch import nn
        import onnx
    except ImportError as exc:
        raise RuntimeError('Vision worker requires torch, numpy, Pillow and onnx in the selected Python environment') from exc
    torch.set_num_threads(int(config.get('threads',2)))
    seed=int(config.get('seed',42));torch.manual_seed(seed);random.seed(seed)
    root=Path(config['dataset_path']);out=Path(config['output_dir'])
    size=int(config.get('image_size',64));batch=int(config.get('batch_size',8))
    if config.get('model','tiny-cnn') not in ('tiny-cnn',''):raise ValueError('Built-in vision architecture is tiny-cnn; import other ONNX models directly for compilation')
    explicit=any((root/subset).is_dir() for subset in ('train','validation','test'))
    classroot=root/'train' if explicit else root
    classes=validate_explicit_splits(root) if explicit else sorted(p.name for p in classroot.iterdir() if p.is_dir())
    if len(classes)<2:raise ValueError('Classification requires at least two class directories')
    train=[];validation=[];test=[];seen=set();split=[]
    for label,name in enumerate(classes):
        if explicit:
            groups=[]
            for subset in ('train','validation','test'):
                folder=root/subset/name
                files=sorted(p for p in folder.rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp') and p.is_file())
                if not files:raise ValueError(f'Missing {subset} images for class {name}')
                groups.extend((p,subset) for p in files)
        else:
            files=sorted(p for p in (root/name).rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp') and p.is_file())
            if len(files)<5:raise ValueError(f'Class {name} needs at least 5 images')
            random.shuffle(files);n=max(1,int(len(files)*0.2))
            groups=[(p,'test' if index<n else 'validation' if index<2*n else 'train') for index,p in enumerate(files)]
        for p,subset in groups:
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            if digest in seen:raise ValueError('Duplicate image content detected; remove duplicates to prevent train/test leakage')
            seen.add(digest)
            if subset=='test':test.append((p,label))
            elif subset=='train':train.append((p,label))
            elif subset=='validation':validation.append((p,label))
            split.append({'path':str(p.relative_to(root)),'sha256':digest,'split':subset})
    if len(train)<4 or len(test)<2:raise ValueError('Need at least four training images and two test images')
    if len(train)+len(validation)+len(test)>100000:raise ValueError('Built-in trainer is limited to 100000 images')
    class Dataset(torch.utils.data.Dataset):
        def __init__(self,records):self.records=records
        def __len__(self):return len(self.records)
        def __getitem__(self,i):
            p,label=self.records[i]
            with Image.open(p) as im:arr=np.asarray(im.convert('RGB').resize((size,size)),dtype=np.float32).copy()/255.0
            return torch.from_numpy(arr.transpose(2,0,1)),label
    model=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(32,len(classes)))
    device=config.get('device','cpu')
    if device not in ('cpu','cuda','mps'):raise ValueError('device must be cpu, cuda or mps')
    model.to(device);optim=torch.optim.Adam(model.parameters(),lr=float(config.get('learning_rate',0.001)))
    loader=torch.utils.data.DataLoader(Dataset(train),batch_size=batch,shuffle=True,num_workers=0)
    start=time.monotonic();history=[]
    for epoch in range(int(config.get('epochs',1))):
        model.train();total=0.0
        for x,y in loader:
            x=x.to(device);y=y.to(device);optim.zero_grad();loss=nn.functional.cross_entropy(model(x),y);loss.backward();optim.step();total+=loss.item()*len(y)
        history.append({'epoch':epoch+1,'train_loss':total/len(train)})
        print(json.dumps(history[-1]),flush=True)
    model.eval();correct=0;total_loss=0.0;confusion=[[0]*len(classes) for _ in classes]
    with torch.no_grad():
        for x,y in torch.utils.data.DataLoader(Dataset(test),batch_size=batch):
            pred=model(x.to(device));total_loss+=nn.functional.cross_entropy(pred,y.to(device),reduction='sum').item()
            labels=pred.argmax(1).cpu();correct+=int((labels==y).sum())
            for actual,guess in zip(y.tolist(),labels.tolist()):confusion[actual][guess]+=1
    validation_metrics={}
    if validation:
        validation_correct=0;validation_loss=0.0
        with torch.no_grad():
            for x,y in torch.utils.data.DataLoader(Dataset(validation),batch_size=batch):
                prediction=model(x.to(device));validation_loss+=nn.functional.cross_entropy(prediction,y.to(device),reduction='sum').item()
                validation_correct+=int((prediction.argmax(1).cpu()==y).sum())
        validation_metrics={'validation_accuracy':validation_correct/len(validation),'validation_loss':validation_loss/len(validation),'validation_count':len(validation)}
    model.cpu();torch.save({'state_dict':model.state_dict(),'classes':classes,'image_size':size,'architecture':'tiny-cnn'},out/'model.pt')
    torch.onnx.export(model,torch.zeros(1,3,size,size),str(out/'model.onnx'),input_names=['images'],output_names=['logits'],opset_version=13,dynamo=False)
    onnx.checker.check_model(onnx.load(str(out/'model.onnx')))
    # Calibration comes exclusively from training images, in DFC's NHWC layout.
    calibration=np.stack([Dataset(train)[i][0].numpy().transpose(1,2,0) for i in range(min(256,len(train)))])
    np.save(out/'calibration.npy',calibration)
    (out/'labels.json').write_text(json.dumps(classes))
    (out/'dataset-split.json').write_text(json.dumps(split,indent=2))
    preprocessing={'input':'images','layout':'NCHW','image_size':size,'colour':'RGB','scale':1/255,'resize':'Pillow bicubic','output':'logits','calibration_layout':'NHWC','calibration_scale':'already float32 0..1'}
    (out/'preprocessing.json').write_text(json.dumps(preprocessing,indent=2))
    result={'artifact_paths':[str(out/p) for p in ('model.pt','model.onnx','calibration.npy','labels.json','preprocessing.json','dataset-split.json')],
            'metrics':{**validation_metrics,'test_accuracy':correct/len(test),'test_loss':total_loss/len(test),'test_count':len(test),'train_count':len(train),'confusion_matrix':confusion,'training_seconds':time.monotonic()-start,'parameters':sum(p.numel() for p in model.parameters())},'history':history,
            'limitations':['Random stratified image holdout; user must prevent subject/session leakage. Test set is evaluated once after training; no hyperparameter selection is performed.','ONNX structural validation does not establish DFC operator compatibility or quantized accuracy.']}
    (out/'training-result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
if __name__=='__main__':main(json.loads(Path(sys.argv[1]).read_text()))
