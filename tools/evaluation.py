"""D4 evaluation: historical 24-view multiscale or 8-view single-scale TTA."""
import hashlib
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import ttach as tta
from torch import nn
from tools.metric import Metrics

CLASSES = ["ImSurf", "Building", "LowVeg", "Tree", "Car", "Clutter"]

def segmentation_scores(matrix):
    m=matrix.double().cpu()
    tp=m.diag()
    actual,predicted=m.sum(1),m.sum(0)
    def divide(a,b):
        return torch.where(b>0,a/b,torch.nan)
    iou=divide(tp,actual+predicted-tp)
    precision=divide(tp,predicted)
    recall=divide(tp,actual)
    f1=divide(2*tp,actual+predicted)
    def scalar(x):
        return x.item() if torch.isfinite(x) else None
    total=m.sum()
    oa=divide(tp.sum(),total)
    chance=divide((actual*predicted).sum(),total.square())
    result=dict(OA=scalar(oa),mIoU=scalar(iou[:5].nanmean()),
                mF1=scalar(f1[:5].nanmean()),mDice=scalar(f1[:5].nanmean()),
                mPrecision=scalar(precision[:5].nanmean()),mRecall=scalar(recall[:5].nanmean()),
                mAcc=scalar(recall[:5].nanmean()),
                mIoU6=scalar(iou.nanmean()),mF1_6=scalar(f1.nanmean()),
                FWIoU=scalar((actual/total*iou.nan_to_num()).sum()),
                Kappa=scalar(divide(oa-chance,1-chance)),
                confusion_matrix=m.long().tolist(),per_class=[])
    for i,name in enumerate(CLASSES):
        result["per_class"].append(dict(name=name,IoU=scalar(iou[i]),F1=scalar(f1[i]),
                                       Dice=scalar(f1[i]),Precision=scalar(precision[i]),
                                       Recall=scalar(recall[i]),support=int(actual[i])))
    return result

class Logits(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model=model
    def forward(self,x):
        return self.model(x)["logits"]

@torch.inference_mode()
def evaluate_tta(model,loader,progress=None,max_batches=None,prediction_dir=None,scales=(.75,1.,1.5)):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    if prediction_dir is not None:
        prediction_dir = Path(prediction_dir)
        prediction_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    operations=[tta.HorizontalFlip(),tta.VerticalFlip(),tta.Rotate90(angles=[90])]
    if scales is not None:
        operations.append(tta.Scale(scales=list(scales),interpolation="bilinear",align_corners=False))
    transforms=tta.Compose(operations)
    assert len(transforms)==8*(len(scales) if scales is not None else 1)
    wrapper=tta.SegmentationTTAWrapper(Logits(model),transforms,merge_mode="mean")
    metrics=Metrics("cuda")
    images=0
    for i,batch in enumerate(loader):
        logits=wrapper(batch["img"].cuda(non_blocking=True))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite TTA logits")
        metrics.update(logits,batch["gt_semantic_seg"].cuda(non_blocking=True))
        if prediction_dir is not None:
            palette = [255,255,255, 0,0,255, 0,255,255, 0,255,0, 255,204,0, 255,0,0]
            palette += [0] * (768-len(palette))
            for prediction, image_id in zip(logits.argmax(1).byte().cpu().numpy(), batch['img_id']):
                name = Path(str(image_id)).name + '.png'
                mask = Image.fromarray(prediction)
                mask.putpalette(palette)
                temporary = prediction_dir / ('.'+name+'.tmp')
                mask.save(temporary, format='PNG')
                temporary.replace(prediction_dir / name)
        images+=logits.shape[0]
        if progress:
            progress(tta_images=images,tta_batches=i+1)
        if max_batches and i+1>=max_batches:
            break
    result=segmentation_scores(metrics.matrix)
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    result.update(evaluation_seconds=seconds, images_per_second=images/seconds,
                  milliseconds_per_image=1000*seconds/images,
                  peak_cuda_memory_mib=torch.cuda.max_memory_allocated()/1024**2)
    result.update(images=images,transforms=len(transforms),scales=list(scales) if scales is not None else [1.],merge="mean logits",
                  macro_classes=CLASSES[:5],ignore_label=6,saved_prediction_images=prediction_dir is not None,
                  prediction_directory=str(prediction_dir) if prediction_dir is not None else None,
                  prediction_format="palette PNG; pixel IDs 0..5" if prediction_dir is not None else None)
    return result


@torch.inference_mode()
def evaluate_plain(model, loader):
    """Best-checkpoint evaluation without test-time augmentation, same pixel protocol."""
    model.eval()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    metrics = Metrics('cuda')
    images = 0
    for batch in loader:
        logits = model(batch['img'].cuda(non_blocking=True))['logits']
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Non-finite plain evaluation logits')
        metrics.update(logits, batch['gt_semantic_seg'].cuda(non_blocking=True))
        images += logits.shape[0]
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    result = segmentation_scores(metrics.matrix)
    result.update(images=images, transforms=1, evaluation_seconds=seconds,
                  images_per_second=images/seconds, milliseconds_per_image=1000*seconds/images,
                  peak_cuda_memory_mib=torch.cuda.max_memory_allocated()/1024**2)
    return result

def sha256(path):
    digest=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()
