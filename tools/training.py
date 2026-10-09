"""One process, one CUDA device, one model with best-weight saving."""
import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from catalyst.contrib.optimizers import Lookahead
from geoseg.models.SFFNet.SFFNet import SFFNet
VARIANTS = {"c10_dino_base": "C10 DINOv3 Base"}
BACKBONES = {"c10_dino_base": "dino_base"}
DINO_PATH = None
from geoseg.datasets.potsdam_dataset import PotsdamDataset, train_aug, val_aug
from geoseg.losses.segmentation import compute_loss
from tools.metric import Metrics
from tools.reporting import now, write_json, read_json, render_model
from tools.evaluation import evaluate_tta, evaluate_plain, sha256


def seed_worker(worker_id):
    seed = torch.initial_seed()%2**32
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


def datasets(root, dataset_name="potsdamR"):
    train = PotsdamDataset(str(Path(root)/'train'),mode='train',mosaic_ratio=0.25,transform=train_aug)
    val = PotsdamDataset(str(Path(root)/'test'),mode='val',transform=val_aug)
    for dataset in (train,val):
        dataset.img_ids.sort()
        base = Path(dataset.data_root)
        images = {p.stem for p in (base/dataset.img_dir).glob('*.tif')}
        masks = {p.stem for p in (base/dataset.mask_dir).glob('*.png')}
        if images != masks or not images:
            raise ValueError(f'Unpaired/empty data: {base}')
    exclusions = read_json(Path(__file__).resolve().parents[1]/'data_exclusions.json') if dataset_name=='potsdamR' else {'train':{},'test':{}}
    assert not exclusions['test'], 'Test set exclusions are not permitted'
    for dataset,split in [(train,'train'),(val,'test')]:
        excluded = set(exclusions[split])
        assert excluded <= set(dataset.img_ids), 'Exclusion manifest does not match dataset'
        dataset.img_ids = [name for name in dataset.img_ids if name not in excluded]
    return train,val


def save_best_weights(model, destination):
    """Atomically persist model weights so an interrupted write cannot be selected."""
    destination = Path(destination)
    temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.tmp')
    try:
        weights = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        torch.save(weights, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--model',choices=VARIANTS,required=True)
    p.add_argument('--ablation', choices=SFFNet.ABLATIONS, default='none')
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--dataset',choices=['vaihingenR','potsdamR'],required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--pretrained',default=None,help=argparse.SUPPRESS)
    p.add_argument('--dino-pretrained',required=True,help='External pinned DINOv3 Base safetensors, not included')
    p.add_argument('--best-checkpoint',type=Path,default=None,
                   help='Optional path for the best model state_dict')
    p.add_argument('--epochs',type=int,default=105)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--prior-record',type=Path)
    p.add_argument('--val-batch-size',type=int,default=2)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--attempt',type=int,default=1)
    p.add_argument('--smoke',action='store_true',help='Two train/validation batches; never production scores')
    return p


def run(args):
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Expose exactly one GPU using CUDA_VISIBLE_DEVICES')
    if args.epochs<1 or args.batch_size<1:
        raise ValueError('epochs and batch size must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    args.run_dir.mkdir(parents=True,exist_ok=True)
    status_path = args.run_dir/'status.json'
    state = dict(model=args.model,ablation=args.ablation,status='initializing',epoch=0,best_val_miou=None,best_epoch=None,
                 best_checkpoint=None,tta=None,
                 gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),attempt=args.attempt,
                 smoke=args.smoke,started_at=now(),pid=os.getpid())

    def publish(**updates):
        state.update(updates,updated_at=now())
        write_json(status_path,state)

    publish()
    try:
        prior = read_json(args.prior_record,{}) if args.prior_record else {}
        if not args.smoke and Path(prior.get('best_checkpoint') or '').is_file():
            for key in ('best_val_miou','best_epoch','best_attempt','best_metrics','best_checkpoint'):
                state[key] = prior.get(key)
            if prior.get('training_completed'):
                state.update(training_completed=True,epoch=prior['epoch'])
        write_json(args.run_dir/'config.json',dict({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},data_exclusions=read_json(Path(__file__).resolve().parents[1]/'data_exclusions.json') if args.dataset=='potsdamR' else {'train':{},'test':{}}))
        train,val = datasets(args.data_root,args.dataset)
        generator = torch.Generator().manual_seed(args.seed)
        loaders = [DataLoader(ds,batch_size=bs,shuffle=shuffle,drop_last=shuffle,
                              num_workers=args.workers,pin_memory=True,
                              persistent_workers=args.workers>0,worker_init_fn=seed_worker,
                              generator=generator)
                   for ds,bs,shuffle in [(train,args.batch_size,True),(val,args.val_batch_size,False)]]
        model = SFFNet(pretrained=args.dino_pretrained, ablation=args.ablation).cuda()
        groups = [dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('backbone.')],lr=6e-5),
                  dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('backbone.')],lr=6e-4)]
        optimizer = Lookahead(torch.optim.AdamW(groups,weight_decay=0.01))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=15,T_mult=2)
        publish(parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                backbone=BACKBONES[args.model],
                pretrained_path=getattr(model.backbone,'pretrained_path',args.dino_pretrained),
                train_images=len(train),val_images=len(val))
        print(json.dumps(state,ensure_ascii=False),flush=True)
        epochs = 1 if args.smoke else args.epochs
        if state.get('training_completed'):
            epochs = 0
            print('Training already completed; retrying best-checkpoint TTA only.',flush=True)
        best_checkpoint = args.best_checkpoint or args.run_dir/'best_model.pth'
        best_checkpoint.parent.mkdir(parents=True,exist_ok=True)
        metric_file = args.run_dir/'metrics.csv'
        with metric_file.open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=['epoch','train_loss','val_miou','val_miou6',
                                                 'val_f1','val_oa','best_val_miou','seconds']+
                                                 [f'iou_class_{i}' for i in range(6)])
            writer.writeheader()
            for epoch in range(1,epochs+1):
                start = time.monotonic()
                publish(status='training',epoch=epoch,step=0,total_steps=len(loaders[0]))
                model.train()
                total_loss,steps = 0.,0
                for batch in loaders[0]:
                    images = batch['img'].cuda(non_blocking=True)
                    labels = batch['gt_semantic_seg'].cuda(non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    output = model(images)
                    loss = compute_loss(output,labels,args.model)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Non-finite loss at epoch {epoch}, batch {steps}')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.0,error_if_nonfinite=True)
                    optimizer.step()
                    total_loss += loss.item()
                    steps += 1
                    if steps%20==0 or steps==1:
                        publish(step=steps,train_loss=total_loss/steps)
                        print(f'epoch={epoch} step={steps}/{len(loaders[0])} loss={total_loss/steps:.5f}',flush=True)
                    del output,loss
                    if args.smoke and steps==2:
                        break
                scheduler.step(epoch)
                model.eval()
                publish(status='validating')
                metrics = Metrics('cuda')
                with torch.inference_mode():
                    for i,batch in enumerate(loaders[1]):
                        pred = model(batch['img'].cuda(non_blocking=True))['logits']
                        if not torch.isfinite(pred).all():
                            raise FloatingPointError('Non-finite validation logits')
                        metrics.update(pred,batch['gt_semantic_seg'].cuda(non_blocking=True))
                        if args.smoke and i==1:
                            break
                scores = metrics.result()
                if not math.isfinite(scores['val_miou']):
                    raise FloatingPointError('Undefined validation mIoU')
                if not args.smoke:
                    if state['best_val_miou'] is None or scores['val_miou']>state['best_val_miou']:
                        save_best_weights(model,best_checkpoint)
                        state.update(best_val_miou=scores['val_miou'],best_epoch=epoch,
                                     best_attempt=args.attempt,best_metrics=dict(scores),
                                     best_checkpoint=str(best_checkpoint.resolve()))
                else:
                    scores = {'smoke_'+k:v for k,v in scores.items()}
                publish(**scores,train_loss=total_loss/max(steps,1),epoch_seconds=time.monotonic()-start)
                row = {k:state.get(k) for k in writer.fieldnames}
                row['seconds'] = state['epoch_seconds']
                if not args.smoke:
                    row.update({f'iou_class_{i}':v for i,v in enumerate(scores['per_class_iou'])})
                writer.writerow(row)
                f.flush()
                print(json.dumps(state,ensure_ascii=False),flush=True)
        # TTA always uses the validation-selected checkpoint, including a better prior attempt.
        if not args.smoke:
            publish(training_completed=True)
            model.load_state_dict(torch.load(state['best_checkpoint'],map_location='cpu'),strict=True)
        del optimizer, scheduler
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        publish(status='testing_tta')
        test_loader = DataLoader(val,batch_size=1,shuffle=False,num_workers=args.workers,
                                 pin_memory=True,worker_init_fn=seed_worker)
        if not args.smoke:
            publish(status='testing_no_tta')
            plain = evaluate_plain(model, test_loader)
            plain.update(ablation=args.ablation, best_epoch=state['best_epoch'],
                         checkpoint_sha256=sha256(state['best_checkpoint']))
            write_json(args.run_dir/'no_tta_metrics.json', plain)
            publish(status='testing_tta')
        result = evaluate_tta(model,test_loader,progress=publish,max_batches=1 if args.smoke else None)
        result.update(model=args.model,ablation=args.ablation,seed=args.seed,split=str(Path(args.data_root)/'test'),
                      best_epoch=state['best_epoch'],best_attempt=state.get('best_attempt'),
                      best_checkpoint=state['best_checkpoint'],smoke=args.smoke)
        if not args.smoke:
            result['checkpoint_sha256'] = sha256(state['best_checkpoint'])
            result['image_ids_sha256'] = __import__('hashlib').sha256('\n'.join(val.img_ids).encode()).hexdigest()
            write_json(args.run_dir/'tta_metrics.json',result)
            publish(tta=result)
        else:
            write_json(args.run_dir/'smoke_tta_metrics.json',result)
        publish(status='smoke_passed' if args.smoke else 'completed',finished_at=now())
        if not args.smoke:
            render_model(state,args.run_dir/'RESULTS.md')
    except BaseException as exc:
        publish(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':
    run(parser().parse_args())
