"""Dataset-level confusion matrix; five-class mIoU follows the original protocol."""
import torch


class Metrics:
    def __init__(self, device='cpu'):
        self.matrix = torch.zeros(6,6,dtype=torch.int64,device=device)

    def update(self, logits, labels):
        pred = logits.argmax(1)
        valid = (labels >= 0)&(labels < 6)
        self.matrix += torch.bincount(6*labels[valid]+pred[valid],minlength=36).reshape(6,6)

    def result(self):
        m = self.matrix.double()
        tp = m.diag()
        union = m.sum(0)+m.sum(1)-tp
        iou = torch.where(union>0,tp/union,torch.nan)
        denom = m.sum(0)+m.sum(1)
        f1 = torch.where(denom>0,2*tp/denom,torch.nan)
        return dict(val_miou=iou[:5].nanmean().item(), val_miou6=iou.nanmean().item(),
                    val_f1=f1[:5].nanmean().item(), val_oa=(tp.sum()/m.sum().clamp_min(1)).item(),
                    per_class_iou=[None if torch.isnan(v) else v.item() for v in iou])
