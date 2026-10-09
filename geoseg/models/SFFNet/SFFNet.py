"""C10 DINOv3 Base segmentation model, consolidated without experiment inheritance.

State-dict keys and forward operations match both packaged best checkpoints.
"""
import math
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import timm
from safetensors.torch import load_file

def resize(x, target):
    size = target.shape[-2:] if isinstance(target, torch.Tensor) else target
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class Conv(nn.Sequential):
    def __init__(self, ci, co, k=3, dilation=1, groups=1):
        super().__init__(nn.Conv2d(ci, co, k, padding=dilation * (k // 2),
                                  dilation=dilation, groups=groups, bias=False),
                         nn.GroupNorm(math.gcd(8, co), co), nn.GELU())

class Residual(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.block = nn.Sequential(Conv(c, c, 3, dilation, c), Conv(c, c, 1))

    def forward(self, x):
        return x + self.block(x)

class Context(nn.Module):
    def __init__(self, c, rates=(1, 3, 5)):
        super().__init__()
        self.branches = nn.ModuleList([Conv(c, c, 3, r, c) for r in rates])
        self.pool = Conv(c, c, 1)
        self.fuse = Conv(c*(len(rates)+1), c, 1)

    def forward(self, x):
        parts = [m(x) for m in self.branches]
        parts.append(resize(self.pool(F.adaptive_avg_pool2d(x, 1)), x))
        return self.fuse(torch.cat(parts, 1))

class GRN(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1,c,1,1))
        self.beta = nn.Parameter(torch.zeros(1,c,1,1))

    def forward(self,x):
        response = x.square().mean((2,3),keepdim=True).add(1e-6).sqrt()
        normalized = response/response.mean(1,keepdim=True).clamp_min(1e-6)
        return x+self.gamma*x*normalized+self.beta

class RestorationAdapter(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = min(64, c // 2)
        self.reduce = Conv(c, d, 1)
        self.qkv = nn.Sequential(nn.Conv2d(d, 3*d, 1), nn.Conv2d(3*d, 3*d, 3, padding=1, groups=3*d))
        self.temperature = nn.Parameter(torch.ones(1))
        self.operator = nn.Sequential(Conv(d, d, 1), GRN(d))
        self.project = nn.Conv2d(d, c, 1)
        self.strength = nn.Parameter(torch.tensor(math.atanh(0.1)))

    def forward(self, x):
        z = self.reduce(x)
        q, k, v = self.qkv(z).chunk(3, 1)
        q, k = [F.normalize(t.flatten(2), dim=-1, eps=1e-6) for t in (q, k)]
        attention = ((0.1 + F.softplus(self.temperature)) * (q @ k.transpose(1, 2))).softmax(-1)
        z = self.operator((attention @ v.flatten(2)).reshape_as(z))
        return x + 0.5 * self.strength.tanh() * self.project(z)


class DinoBackbone(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        self.encoder = timm.create_model('convnext_base.dinov3_lvd1689m', pretrained=False,
                                         num_classes=0, drop_path_rate=0.0)
        self.pretrained_path = None
        if pretrained:
            path = Path(pretrained)
            expected = 'afa69ae6094b5b04a48810de78c52c92d4cbb6d9436462f07bf36defb09e26c1'
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Backbone SHA256 mismatch: {path}')
            self.encoder.load_state_dict(load_file(str(path), device='cpu'), strict=True)
            self.pretrained_path = str(path)
        self.channels = tuple(info['num_chs'] for info in self.encoder.feature_info)
        self.final_norm = self.encoder.head.norm
        self.encoder.head = nn.Identity()
        self.adapters = nn.ModuleList([nn.Identity() for _ in self.channels])

    def configure(self):
        self.adapters = nn.ModuleList([RestorationAdapter(c) for c in self.channels])

    def forward_multiscale(self, x):
        x = self.encoder.stem(x)
        outputs = []
        for i, (stage, adapter) in enumerate(zip(self.encoder.stages, self.adapters)):
            x = adapter(stage(x))
            outputs.append(self.final_norm(x) if i == 3 else x)
        return tuple(outputs)


def entropy(logits):
    p = logits.softmax(1)
    return -(p*p.clamp_min(1e-6).log()).sum(1,keepdim=True)/math.log(logits.shape[1])

class ObjectContext(nn.Module):
    def __init__(self,c,k):
        super().__init__()
        self.region = nn.Conv2d(c,k,1)
        self.query = nn.Conv2d(c,c,1)
        self.key = nn.Linear(c,c)
        self.value = nn.Linear(c,c)
        self.fuse = nn.Sequential(Conv(2*c,c),Residual(c))

    def forward(self,x):
        region = self.region(x)
        assignment = region.flatten(2).softmax(-1)
        prototypes = assignment @ x.flatten(2).transpose(1,2)
        q = F.normalize(self.query(x).flatten(2).transpose(1,2),dim=-1)
        k = F.normalize(self.key(prototypes),dim=-1)
        affinity = (8*(q @ k.transpose(1,2))).softmax(-1)
        context = (affinity @ self.value(prototypes)).transpose(1,2).reshape_as(x)
        return self.fuse(torch.cat([x,context],1)),region

class ReverseDecoder(nn.Module):
    """V07's supervised coarse-to-fine residual decoder, reusable with new features."""
    def __init__(self,c,k):
        super().__init__()
        self.context = Context(c)
        self.coarse = nn.Conv2d(c,k,1)
        self.refine = nn.ModuleList([Conv(2*c+k,c) for _ in range(3)])
        self.correct = nn.ModuleList([nn.Conv2d(c,k,1) for _ in range(3)])

    def forward(self,p):
        z = self.context(p[-1])
        logits = self.coarse(z)
        aux = [logits]
        for i in (2,1,0):
            logits = resize(logits,p[i])
            reverse = 1-logits.softmax(1).amax(1,keepdim=True)
            z = self.refine[i](torch.cat([p[i]*(1+reverse),resize(z,p[i]),logits],1))
            logits = logits+self.correct[i](z)
            if i:
                aux.append(logits)
        return z,logits,aux

class PrototypeExchange(nn.Module):
    def __init__(self,c,k):
        super().__init__()
        self.assign = nn.Conv2d(c,k,1)
        self.mix = Conv(2*c,c,1)
        self.temperature = nn.Parameter(torch.tensor(2.))
    def forward(self,fine,deep):
        regions = self.assign(deep)
        prototypes = regions.flatten(2).softmax(-1) @ deep.flatten(2).transpose(1,2)
        logits = F.normalize(fine.flatten(2),dim=1).transpose(1,2) @ F.normalize(prototypes,dim=-1).transpose(1,2)
        attention = (logits*(1+F.softplus(self.temperature))).softmax(-1)
        context = (attention@prototypes).transpose(1,2).reshape_as(fine)
        return fine+.1*self.mix(torch.cat([fine,context],1)),regions

class GroupedSampler(nn.Module):
    """DySample-inspired grouped point sampling; arbitrary target size, bounded offsets."""
    def __init__(self,c,groups=4):
        super().__init__()
        self.groups=groups
        self.offset=nn.Conv2d(c,2*groups,1)
        nn.init.normal_(self.offset.weight,std=.001)
        nn.init.zeros_(self.offset.bias)
        self.mix=Conv(2*c,c,1)
    def forward(self,fine,coarse):
        b,c,h,w=coarse.shape
        th,tw=fine.shape[-2:]
        offset=resize(self.offset(coarse),fine).reshape(b,self.groups,2,th,tw).permute(0,1,3,4,2)
        y,x=torch.meshgrid(torch.arange(th,device=fine.device,dtype=fine.dtype),
                           torch.arange(tw,device=fine.device,dtype=fine.dtype),indexing="ij")
        grid=torch.stack([2*(x+.5)/tw-1,2*(y+.5)/th-1],-1)
        grid=grid[None,None]+offset.tanh()*offset.new_tensor([1/w,1/h])
        aligned=F.grid_sample(coarse.reshape(b*self.groups,c//self.groups,h,w),
                              grid.reshape(b*self.groups,th,tw,2),padding_mode="border",align_corners=False)
        aligned=aligned.reshape(b,c,th,tw)
        return fine+.1*self.mix(torch.cat([fine,aligned],1))

class SFFNet(nn.Module):
    """C10 implementation under the original SFFNet import path."""
    ABLATIONS = ('none', 'no_adapter', 'no_sampling', 'no_scale_route',
                 'no_exchange', 'no_reverse', 'no_detail')

    def __init__(self, num_classes=6, width=96, pretrained=None, ablation='none'):
        super().__init__()
        if ablation not in self.ABLATIONS:
            raise ValueError(f'Unknown ablation: {ablation}')
        self.ablation = ablation
        self.width, self.num_classes = width, num_classes
        c, k = width, num_classes
        self.backbone = DinoBackbone(pretrained)
        self.projections = nn.ModuleList([Conv(cin, c, 1) for cin in self.backbone.channels])
        # Preserve the historical initialization sequence before removing unused heads.
        self.head = nn.Sequential(Residual(c), nn.Dropout2d(0.1), nn.Conv2d(c, k, 1))
        self.aux_head = nn.Conv2d(c, k, 1)
        self.head = nn.Identity()
        self.aux_head = nn.Identity()
        self.deep_context = ObjectContext(c, k)
        self.reverse = ReverseDecoder(c, k)
        self.fine_context = ObjectContext(c, k)
        self.residual = nn.Sequential(Conv(2*c, c), nn.Conv2d(c, k, 1))
        self.boundary = nn.Conv2d(c, 1, 1)
        self.detail = nn.Sequential(Conv(c+2, c), Residual(c))
        self.error = nn.Conv2d(c, 1, 1)
        self.car = nn.Conv2d(c, 1, 1)
        self.correction = nn.Conv2d(c, k, 1)
        self.backbone.configure()
        self.exchange = nn.ModuleList([PrototypeExchange(c, k), PrototypeExchange(c, k)])
        self.scale_route = nn.Sequential(Conv(4*c, c, 1), nn.Conv2d(c, 4, 1))
        self.route_mix = Conv(c, c, 1)
        self.sampling = nn.ModuleList([GroupedSampler(c), GroupedSampler(c)])
        # Remove after initialization so shared parameters retain seed-matched values.
        if ablation == 'no_adapter':
            self.backbone.adapters = nn.ModuleList([nn.Identity() for _ in self.backbone.channels])
        elif ablation == 'no_sampling':
            self.sampling = nn.ModuleList()
        elif ablation == 'no_scale_route':
            self.scale_route = self.route_mix = nn.Identity()
        elif ablation == 'no_exchange':
            self.exchange = nn.ModuleList()
        elif ablation == 'no_reverse':
            self.reverse = PlainDecoder(c, k)
        elif ablation == 'no_detail':
            self.detail = self.error = self.car = self.correction = nn.Identity()

    def decode(self, features):
        p = list(features)
        if self.ablation != 'no_sampling':
            for i in (1, 0):
                p[i] = self.sampling[i](p[i], p[i+1])
        if self.ablation != 'no_scale_route':
            scales = [resize(t, p[0]) for t in p]
            weights = self.scale_route(torch.cat(scales, 1)).softmax(1)
            p[0] = p[0] + .1 * self.route_mix(sum(weights[:, i:i+1]*t for i, t in enumerate(scales)))
        exchange_aux = []
        if self.ablation != 'no_exchange':
            p[0], a = self.exchange[0](p[0], p[3])
            p[3], b = self.exchange[1](p[3], F.adaptive_avg_pool2d(p[0], p[3].shape[-2:]))
            exchange_aux = [a, b]
        fine = p[0]
        p[3], deep_region = self.deep_context(p[3])
        z, coarse, aux = self.reverse(p)
        refined, region = self.fine_context(z)
        delta = self.residual(torch.cat([z, refined], 1))
        coarse = coarse + (0.5 + 0.5*entropy(coarse))*delta
        boundary = self.boundary(refined)
        if self.ablation == 'no_detail':
            return dict(logits=coarse, aux=aux + [deep_region, region] + exchange_aux,
                        boundary=boundary)
        top = coarse.softmax(1).topk(2, dim=1).values
        margin = 1 - (top[:, 0:1] - top[:, 1:2])
        detail = self.detail(torch.cat([fine, margin, entropy(coarse)], 1))
        error = self.error(detail)
        logits = coarse + 0.5*(margin + error.sigmoid())*self.correction(detail)
        return dict(logits=logits, aux=aux + [deep_region, region] + exchange_aux, boundary=boundary,
                    errors=[(error, coarse)], car=self.car(detail))

    def forward(self, x):
        features = [projection(feature) for projection, feature in
                    zip(self.projections, self.backbone.forward_multiscale(x))]
        result = self.decode(features)
        result['logits'] = resize(result['logits'], x)
        return result


class PlainDecoder(nn.Module):
    """Control: ordinary top-down fusion, no reverse attention or logit correction."""
    def __init__(self, c, k):
        super().__init__()
        self.context = Context(c)
        self.fuse = nn.ModuleList([Conv(2*c, c) for _ in range(3)])
        self.heads = nn.ModuleList([nn.Conv2d(c, k, 1) for _ in range(4)])

    def forward(self, p):
        z = self.context(p[-1])
        aux = [self.heads[3](z)]
        for i in (2, 1, 0):
            z = self.fuse[i](torch.cat([p[i], resize(z, p[i])], 1))
            logits = self.heads[i](z)
            if i:
                aux.append(logits)
        return z, logits, aux


C10DinoBase = SFFNet
