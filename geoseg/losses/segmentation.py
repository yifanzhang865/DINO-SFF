"""Ignore-safe region, side-output, edge and small-object supervision."""
import torch
from torch.nn import functional as F
from geoseg.models.SFFNet.SFFNet import resize


def segmentation_loss(logits, labels, hard=False):
    logits = resize(logits.float(), labels.shape[-2:])
    valid = (labels >= 0) & (labels < logits.shape[1])
    if not valid.any():
        return logits.sum()*0
    clean = labels.masked_fill(~valid, 6)
    ce = F.cross_entropy(logits, clean, ignore_index=6, label_smoothing=0.05, reduction='none')[valid]
    if hard:
        ce = ce.topk(max(1,ce.numel()//4)).values
    prob = logits.softmax(1)*valid[:,None]
    onehot = F.one_hot(labels.clamp(0,logits.shape[1]-1),logits.shape[1]).permute(0,3,1,2)
    onehot = onehot*valid[:,None]
    inter = (prob*onehot).sum((0,2,3))
    denom = (prob+onehot).sum((0,2,3))
    present = onehot.sum((0,2,3)) > 0
    dice = (1-(2*inter+0.05)/(denom+0.05))[present].mean()
    return ce.mean()+dice


def boundary_target(labels):
    valid = (labels >= 0) & (labels < 6)
    edge = torch.zeros_like(valid)
    dy = (labels[:,1:] != labels[:,:-1]) & valid[:,1:] & valid[:,:-1]
    dx = (labels[:,:,1:] != labels[:,:,:-1]) & valid[:,:,1:] & valid[:,:,:-1]
    edge[:,1:] |= dy
    edge[:,:-1] |= dy
    edge[:,:,1:] |= dx
    edge[:,:,:-1] |= dx
    edge = F.max_pool2d(edge[:,None].float(),3,1,1)[:,0]
    # Ignore a one-pixel band around ignored labels as well.
    safe = F.max_pool2d((~valid)[:,None].float(),3,1,1)[:,0] == 0
    return edge, safe


def binary_loss(logits, target, valid):
    z = resize(logits.float(),target.shape[-2:])[:,0]
    if not valid.any():
        return z.sum()*0
    z,t = z[valid],target[valid].float()
    pos = t.sum()
    weight = ((t.numel()-pos)/(pos+1)).clamp(1,20)
    bce = F.binary_cross_entropy_with_logits(z,t,pos_weight=weight)
    p = z.sigmoid()
    dice = 1-(2*(p*t).sum()+1)/(p.sum()+t.sum()+1)
    return bce+dice


def compute_loss(output, labels, variant):
    # Fourth-round variants retain the full-pixel CE + Dice training protocol.
    loss = segmentation_loss(output['logits'],labels)
    aux = output.get('aux',[])
    if aux:
        loss = loss+0.4*sum(segmentation_loss(x,labels) for x in aux)/len(aux)
    if 'boundary' in output:
        target,valid = boundary_target(labels)
        loss = loss+0.2*binary_loss(output['boundary'],target,valid)
        edge_aux = output.get('boundary_aux',[])
        if edge_aux:
            loss = loss+0.1*sum(binary_loss(z,target,valid) for z in edge_aux)/len(edge_aux)
    if 'body' in output:
        edge,safe = boundary_target(labels)
        body_labels = labels.masked_fill((edge>0)|~safe,6)
        loss = loss+0.2*segmentation_loss(output['body'],body_labels)
    if 'errors' in output:
        valid = (labels>=0)&(labels<6)
        error_losses = []
        for error,coarse in output['errors']:
            # Labels never enter the forward pass; detach the training target.
            predicted = resize(coarse.detach(),labels.shape[-2:]).argmax(1)
            error_losses.append(binary_loss(error,predicted!=labels,valid))
        loss = loss+0.1*sum(error_losses)/len(error_losses)
    if 'car' in output:
        loss = loss+0.2*binary_loss(output['car'],labels==4,(labels>=0)&(labels<6))
    if 'vegetation' in output:
        logits = resize(output['vegetation'].float(),labels.shape[-2:])
        valid = (labels==2)|(labels==3)
        if valid.any():
            target = (labels-2).masked_fill(~valid,-100)
            loss = loss+0.1*F.cross_entropy(logits,target,ignore_index=-100)
        else:
            loss = loss+logits.sum()*0
    if 'balance' in output:
        loss = loss+0.01*output['balance']
    if 'embedding' in output:
        loss = loss+0.05*prototype_metric_loss(output['embedding'],labels)
    return loss


def prototype_metric_loss(embedding,labels):
    """Disjoint support/query samples, capped at 64 per class; no memory bank."""
    small = F.interpolate(labels[:,None].float(),size=embedding.shape[-2:],mode='nearest')[:,0].long()
    features = F.normalize(embedding.float().permute(0,2,3,1).reshape(-1,embedding.shape[1]),dim=-1)
    targets = small.flatten()
    prototypes,queries,query_labels = [],[],[]
    for category in range(6):
        ids = (targets==category).nonzero(as_tuple=True)[0]
        if ids.numel()<4:
            continue
        positions = torch.linspace(0,ids.numel()-1,min(64,ids.numel()),device=ids.device).long()
        samples = features[ids[positions]]
        prototypes.append(samples[::2].mean(0))
        queries.append(samples[1::2])
        query_labels.append(torch.full((samples[1::2].shape[0],),len(prototypes)-1,
                                       device=ids.device,dtype=torch.long))
    if len(prototypes)<2:
        return embedding.sum()*0
    prototypes = F.normalize(torch.stack(prototypes),dim=-1)
    logits = torch.cat(queries)@prototypes.T/0.2
    return F.cross_entropy(logits,torch.cat(query_labels))
