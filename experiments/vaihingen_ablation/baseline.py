"""Reevaluate the packaged baseline on the same dataset and evaluation code."""
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from geoseg.models.SFFNet.SFFNet import SFFNet
from tools.training import datasets, seed_worker
from tools.evaluation import evaluate_plain, evaluate_tta, sha256
from tools.reporting import write_json, now

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/vaihingen_ablation/baseline'
DATA=Path('/data2/tangyangpu/WorkSpace/RS/datasets/vaihingenR')

def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    checkpoint=ROOT/'checkpoints/c10_dino_base_vaihingenR_best.pth'
    entry=json.loads((ROOT/'manifest.json').read_text())['vaihingenR']
    assert sha256(checkpoint)==entry['sha256']
    model=SFFNet()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    model.cuda().eval()
    _,val=datasets(DATA,'vaihingenR')
    state=dict(status='testing_no_tta',model='c10_dino_base',ablation='none',epoch=105,
               best_epoch=entry['best_epoch'],best_checkpoint=str(checkpoint),
               parameters=sum(p.numel() for p in model.parameters()),started_at=now())
    def publish(**updates):
        state.update(updates,updated_at=now())
        write_json(OUT/'status.json',state)
    publish()
    loader=DataLoader(val,batch_size=1,shuffle=False,num_workers=4,pin_memory=True,worker_init_fn=seed_worker)
    meta=dict(checkpoint_sha256=entry['sha256'],best_epoch=entry['best_epoch'],seed=42,
              image_ids_sha256=hashlib.sha256('\n'.join(val.img_ids).encode()).hexdigest(),
              best_checkpoint=str(checkpoint),split=str(DATA/'test'))
    plain=evaluate_plain(model,loader)
    plain.update(meta)
    write_json(OUT/'no_tta_metrics.json',plain)
    publish(status='testing_tta')
    tta=evaluate_tta(model,loader,progress=publish)
    tta.update(meta)
    write_json(OUT/'tta_metrics.json',tta)
    archived=json.loads((ROOT/'results/vaihingenR/tta_metrics.json').read_text())
    publish(status='completed',finished_at=now(),tta=tta,
            archive_confusion_matrix_matches=tta['confusion_matrix']==archived['confusion_matrix'])

if __name__=='__main__':
    main()
