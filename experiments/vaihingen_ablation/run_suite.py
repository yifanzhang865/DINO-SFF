"""Run isolated experiments on GPUs 3..7; refresh Markdown until all finish."""
import fcntl
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from experiments.vaihingen_ablation.report import ROOT, OUT, VARIANTS, render, read
from tools.reporting import write_json, now

PRETRAINED=Path('/data2/tangyangpu/WorkSpace/RS/pretrained/round12/convnext_base.dinov3_lvd1689m/model.safetensors')
DATA=Path('/data2/tangyangpu/WorkSpace/RS/datasets/vaihingenR')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'suite.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert hashlib.sha256(PRETRAINED.read_bytes()).hexdigest()=='afa69ae6094b5b04a48810de78c52c92d4cbb6d9436462f07bf36defb09e26c1'
    splits={}
    for split in ['train','test']:
        images=sorted((DATA/split/'images_1024').glob('*.tif'))
        masks=sorted((DATA/split/'masks_1024').glob('*.png'))
        assert images and [p.stem for p in images]==[p.stem for p in masks]
        splits[split]=dict(images=len(images),ids=[p.stem for p in images])
    assert not set(splits['train']['ids']) & set(splits['test']['ids'])
    sources=[p for folder in ['geoseg','tools','experiments/vaihingen_ablation'] for p in (ROOT/folder).rglob('*.py')]
    manifest=dict(started_at=now(),pid=os.getpid(),python=sys.executable,gpus=[3,4,5,6,7],
                  dataset=str(DATA),splits=splits,pretrained=str(PRETRAINED),
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    write_json(OUT/'suite_manifest.json',manifest)
    q=queue.Queue()
    for name in VARIANTS:
        previous=read(OUT/name/'status.json')
        if previous.get('status')=='completed':
            continue
        if (OUT/name/'best_model.pth').exists():
            raise RuntimeError(f'{name}: existing incomplete training; refusing to overwrite best weights')
        write_json(OUT/name/'status.json',dict(status='queued',ablation=name))
        q.put(name)
    def worker(gpu):
        while True:
            try: name=q.get_nowait()
            except queue.Empty: return
            dest=OUT/name
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',
                     MKL_NUM_THREADS='4',MPLCONFIGDIR='/tmp/c10_ablation_mpl',PYTHONPATH=str(ROOT))
            if name=='baseline':
                cmd=[sys.executable,'-u','-m','experiments.vaihingen_ablation.baseline']
            else:
                cmd=[sys.executable,'-u','-m','tools.training','--model','c10_dino_base',
                     '--ablation',name,'--dataset','vaihingenR','--data-root',str(DATA),
                     '--dino-pretrained',str(PRETRAINED),'--run-dir',str(dest),
                     '--epochs','105','--batch-size','4','--val-batch-size','2','--workers','4','--seed','42']
            write_json(dest/'launch.json',dict(command=cmd,gpu=gpu,started_at=now()))
            print(f'{now()} GPU {gpu} start {name}',flush=True)
            try:
                with (dest/'train.log').open('a') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    write_json(dest/'process.json',dict(pid=child.pid,gpu=gpu))
                    code=child.wait()
                if code:
                    state=read(dest/'status.json')
                    state.update(status='failed',returncode=code,finished_at=now())
                    write_json(dest/'status.json',state)
                print(f'{now()} GPU {gpu} finish {name}: exit {code}',flush=True)
            except Exception as e:
                write_json(dest/'status.json',dict(status='failed',error=repr(e)))
            finally:
                q.task_done()
    threads=[threading.Thread(target=worker,args=(gpu,)) for gpu in [3,4,5,6,7]]
    for t in threads: t.start()
    while any(t.is_alive() for t in threads):
        render()
        time.sleep(30)
    for t in threads: t.join()
    render()
    states={name:read(OUT/name/'status.json').get('status') for name in VARIANTS}
    write_json(OUT/'suite_status.json',dict(status='completed' if all(v=='completed' for v in states.values()) else 'failed',finished_at=now(),experiments=states))
    if any(v!='completed' for v in states.values()):
        raise SystemExit(1)

if __name__=='__main__':
    main()
