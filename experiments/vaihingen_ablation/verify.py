import gc
import json
from pathlib import Path
import torch
from geoseg.models.SFFNet.SFFNet import SFFNet
from geoseg.losses.segmentation import compute_loss
from tools.evaluation import sha256
from tools.reporting import write_json

root = Path(__file__).resolve().parents[2]
torch.set_num_threads(4)
records = []
for ablation in SFFNet.ABLATIONS:
    torch.manual_seed(42)
    model = SFFNet(ablation=ablation).cuda()
    if ablation == 'none':
        path = root/'checkpoints/c10_dino_base_vaihingenR_best.pth'
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        manifest = json.loads((root/'manifest.json').read_text())['vaihingenR']
        assert sha256(path) == manifest['sha256']
    model.train()
    x = torch.randn(2,3,64,64,device='cuda')
    y = torch.randint(0,7,(2,64,64),device='cuda')
    output = model(x)
    assert output['logits'].shape == (2,6,64,64)
    loss = compute_loss(output,y,'c10_dino_base')
    loss.backward()
    assert torch.isfinite(loss)
    bad = [n for n,p in model.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not bad, bad
    record = dict(ablation=ablation, parameters=sum(p.numel() for p in model.parameters()), loss=loss.item(), finite_gradients=True)
    records.append(record)
    print(record,flush=True)
    del model, output, loss
    gc.collect()
    torch.cuda.empty_cache()
write_json(root/'experiments/vaihingen_ablation/verification.json',records)
