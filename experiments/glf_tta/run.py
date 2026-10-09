"""Evaluate both datasets using the local GLF d4 protocol; merge pixel counts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from tools.evaluation import segmentation_scores, sha256
from tools.reporting import write_json, atomic_text, now

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/glf_tta'
DATA=Path('/data2/tangyangpu/WorkSpace/RS/datasets')
GLF=Path('/data2/tangyangpu/WorkSpace/RS/model/GLF-Net-main/GLF-Net-main')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assignments=[('vaihingenR',0,2,3),('vaihingenR',1,2,4),
                 ('potsdamR',0,3,5),('potsdamR',1,3,6),('potsdamR',2,3,7)]
    jobs=[]
    sources={name:dict(path=str(GLF/(name+'_test.py')),sha256=sha256(GLF/(name+'_test.py'))) for name in ['vaihingen','potsdam']}
    write_json(OUT/'status.json',dict(status='testing',started_at=now(),glf_sources=sources))
    for dataset,index,count,gpu in assignments:
        name=dataset[:-1]
        dest=OUT/dataset/f'shard_{index}'
        dest.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,'-u',name+'_test.py','-c',f'config/{name}/sffnet.py',
             '--data-root',str(DATA/dataset),'-t','glf','--no-save','--batch-size','2',
             '--num-shards',str(count),'--shard-index',str(index),'-o',str(dest)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MPLCONFIGDIR='/tmp/c10_glf_mpl')
        log=(dest/'test.log').open('w')
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((p,log,dataset,index))
        write_json(dest/'launch.json',dict(command=cmd,pid=p.pid,gpu=gpu))
    while any(p.poll() is None for p,_,_,_ in jobs):
        progress={f'{d}/{i}':json.loads((OUT/d/f'shard_{i}'/'progress.json').read_text()) if (OUT/d/f'shard_{i}'/'progress.json').exists() else {} for _,_,d,i in jobs}
        write_json(OUT/'progress.json',progress)
        time.sleep(10)
    failed=[]
    for p,log,d,i in jobs:
        log.close()
        if p.returncode: failed.append((d,i,p.returncode))
    if failed:
        write_json(OUT/'status.json',dict(status='failed',failures=failed))
        raise RuntimeError(failed)
    merged={}
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for dataset,count in [('vaihingenR',2),('potsdamR',3)]:
        shards=[json.loads((OUT/dataset/f'shard_{i}'/'tta_metrics.json').read_text()) for i in range(count)]
        expected=sorted(p.stem for p in (DATA/dataset/'test/images_1024').glob('*.tif'))
        ids=[name for s in shards for name in s['image_ids']]
        assert len(set(ids))==len(ids) and sorted(ids)==expected
        assert sum(s['images'] for s in shards)==len(expected)
        assert all(s['checkpoint_sha256']==manifest[dataset]['sha256'] for s in shards)
        matrix=sum((torch.tensor(s['confusion_matrix'],dtype=torch.int64) for s in shards))
        result=segmentation_scores(matrix)
        result.update(images=len(ids),image_ids_sha256=hashlib.sha256('\n'.join(expected).encode()).hexdigest(),
                      checkpoint_sha256=manifest[dataset]['sha256'],transforms=shards[0]['transforms'],
                      scales=shards[0]['scales'],batch_size=2,tta_protocol='local GLF d4',
                      glf_source=sources[dataset[:-1]],best_epoch=manifest[dataset]['best_epoch'])
        archived=json.loads((ROOT/'results'/dataset/'tta_metrics.json').read_text())
        assert [sum(row) for row in matrix.tolist()]==[sum(row) for row in archived['confusion_matrix']]
        result['delta_vs_24view']={k:result[k]-archived[k] for k in ['OA','mIoU','mF1']}
        write_json(OUT/dataset/'tta_metrics.json',result)
        merged[dataset]=result
    lines=['# C10 完整模型：与本地 GLF 一致的 TTA 指标','',f'完成时间：{now()}','',
           '使用两个数据集原有最佳权重，不重新训练。Vaihingen 为 24 路多尺度 D4；Potsdam 为 8 路单尺度 D4。FP32、batch=2；分片测试后累加混淆矩阵，未平均分片指标。','',
           'mIoU/mF1/mPrecision/mRecall/mDice/mAcc 为前五类均值，排除 Clutter；OA/FWIoU/Kappa 为六类。指标统一以百分比显示（Kappa 同样乘100）。沿用 test 集选最佳权重，不是独立盲测。','',
           '| 数据集 | 图数 | TTA | OA | mIoU | mF1 | mPrecision | mRecall | mDice | mAcc | FWIoU | Kappa | mIoU6 | mF1_6 |',
           '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    keys=['OA','mIoU','mF1','mPrecision','mRecall','mDice','mAcc','FWIoU','Kappa','mIoU6','mF1_6']
    for d,r in merged.items():
        lines.append('| '+d+f" | {r['images']} | {r['transforms']} 路 | "+' | '.join(f'{100*r[k]:.4f}' for k in keys)+' |')
    lines+=['','## 与原先统一 24 路 TTA 的差值（百分点）','','| 数据集 | ΔOA | ΔmIoU | ΔmF1 |','|---|---:|---:|---:|']
    for d,r in merged.items():
        lines.append('| '+d+' | '+' | '.join(f'{100*r["delta_vs_24view"][k]:+.4f}' for k in ['OA','mIoU','mF1'])+' |')
    for d,r in merged.items():
        lines+=['',f'## {d} 逐类指标','','| 类别 | IoU | F1/Dice | Precision | Recall | 像素数 |','|---|---:|---:|---:|---:|---:|']
        for c in r['per_class']:
            lines.append('| '+c['name']+' | '+' | '.join(f'{100*c[k]:.4f}' for k in ['IoU','F1','Precision','Recall'])+f" | {c['support']} |")
        lines+=['',f"GLF 源文件：`{r['glf_source']['path']}`；SHA256：`{r['glf_source']['sha256']}`。",f"权重 SHA256：`{r['checkpoint_sha256']}`。",'']
    lines+=['## 复现','','```bash','CUDA_VISIBLE_DEVICES=3 /data2/tangyangpu/RS/bin/python vaihingen_test.py -c config/vaihingen/sffnet.py --data-root /data2/tangyangpu/WorkSpace/RS/datasets/vaihingenR -t glf --batch-size 2 --no-save -o outputs/glf_vaihingen','CUDA_VISIBLE_DEVICES=4 /data2/tangyangpu/RS/bin/python potsdam_test.py -c config/potsdam/sffnet.py --data-root /data2/tangyangpu/WorkSpace/RS/datasets/potsdamR -t glf --batch-size 2 --no-save -o outputs/glf_potsdam','```','']
    atomic_text(OUT/'RESULTS.md','\n'.join(lines))
    write_json(OUT/'status.json',dict(status='completed',finished_at=now(),glf_sources=sources))

if __name__=='__main__':
    main()
