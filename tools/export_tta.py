"""Run full saved-mask TTA using idle GPUs and aggregate disjoint shards."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image
import torch
from tools.evaluation import segmentation_scores
from tools.reporting import write_json, render_model

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tta_d4_results'
DATA=Path('/data2/tangyangpu/WorkSpace/RS/datasets')

def lock(path):
    f=open(path,'a+')
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:f.close();return None
    return f

def idle():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    occupied=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines())
    result=[]
    for line in raw.splitlines():
        index,uuid,used,free,util=[s.strip() for s in line.split(',')]
        if uuid not in occupied and int(used)<=1000 and int(free)>=20000 and int(util)<=5:
            result.append((index,uuid))
    return result

def aggregate(dataset,folder,count):
    directory=OUT/folder
    parts=[json.loads((directory/'.shards'/str(i)/'tta_metrics.json').read_text()) for i in range(count)]
    ids=sorted(name for part in parts for name in part['image_ids'])
    expected=sorted(p.stem for p in (DATA/dataset/'test/images_1024').glob('*.tif'))
    assert ids==expected and len(ids)==len(set(ids))
    entry=json.loads((ROOT/'manifest.json').read_text())[dataset]
    assert all(r['checkpoint_sha256']==entry['sha256'] for r in parts)
    matrix=sum((torch.tensor(r['confusion_matrix'],dtype=torch.int64) for r in parts),torch.zeros(6,6,dtype=torch.int64))
    # Recompute confusion from the exported images, checking size, palette, IDs and completeness.
    assert {p.stem for p in directory.glob('*.png')}==set(ids)
    exported=np.zeros((6,6),dtype=np.int64)
    for name in ids:
        with Image.open(directory/(name+'.png')) as im:
            assert im.mode=='P' and im.size==(1024,1024)
            prediction=np.array(im)
            assert prediction.max()<6
            assert im.getpalette()[:18]==[255,255,255,0,0,255,0,255,255,0,255,0,255,204,0,255,0,0]
        with Image.open(DATA/dataset/'test/masks_1024'/(name+'.png')) as im:target=np.array(im)
        valid=target<6
        exported+=np.bincount(6*target[valid].astype(np.int64)+prediction[valid],minlength=36).reshape(6,6)
    assert np.array_equal(exported,matrix.numpy())
    result=dict(parts[0]);result.update(segmentation_scores(matrix))
    for key in ['shard_index','num_shards','image_ids']:result.pop(key,None)
    result.update(images=len(ids),image_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
                  prediction_directory=str(directory),exported_images=len(ids),saved_masks_confusion_verified=True)
    write_json(directory/'tta_metrics.json',result)
    render_model(dict(model='c10_dino_base',status='completed',best_epoch=entry['best_epoch'],
                     best_val_miou=entry['best_val_miou'],best_checkpoint=str(ROOT/entry['checkpoint']),tta=result),directory/'RESULTS.md')
    return {k:result[k] for k in ['images','OA','mIoU','mF1','saved_masks_confusion_verified']}

def main():
    OUT.mkdir(exist_ok=True)
    singleton=lock(OUT/'.export.lock')
    if singleton is None:raise RuntimeError('Another export is active')
    pending=[('vaihingenR','vaihingen','vaihigen',0,1)]+[('potsdamR','potsdam','postdam',i,4) for i in range(4)]
    active=[];failed=[]
    while pending or active:
        for job in list(active):
            code=job['process'].poll()
            if code is not None:
                job['lease'].close();job['log'].close();active.remove(job)
                print('Finished',job['task'],'exit',code,flush=True)
                if code:failed.append(job['task'])
        for gpu,uuid in idle():
            if not pending:break
            lease=lock(Path('/tmp')/f'glf_variants_{uuid}.lock')
            if lease is None:continue
            if (gpu,uuid) not in idle():lease.close();continue
            task=pending.pop(0);dataset,entry,folder,index,count=task
            out=OUT/folder/'.shards'/str(index);out.mkdir(parents=True,exist_ok=True)
            log=open(out/'test.log','w')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT))
            cmd=[sys.executable,str(ROOT/f'{entry}_test.py'),'-c',str(ROOT/'config'/entry/'sffnet.py'),
                 '--data-root',str(DATA/dataset),'-t','d4','-o',str(out),'--prediction-dir',str(OUT/folder),
                 '--shard-index',str(index),'--num-shards',str(count)]
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,pass_fds=(lease.fileno(),))
            active.append(dict(process=proc,lease=lease,log=log,task=task))
            print('Started',task,'GPU',gpu,'PID',proc.pid,flush=True)
        write_json(OUT/'status.json',dict(status='testing',active=[list(j['task']) for j in active],pending=pending,failed=failed))
        if pending or active:time.sleep(5)
    if failed:
        write_json(OUT/'status.json',dict(status='failed',failed=failed));raise RuntimeError(failed)
    summaries={dataset:aggregate(dataset,folder,count) for dataset,folder,count in [('vaihingenR','vaihigen',1),('potsdamR','postdam',4)]}
    write_json(OUT/'status.json',dict(status='completed',results=summaries))
    print(json.dumps(summaries,indent=2),flush=True)
    singleton.close()

if __name__=='__main__':main()
