from config.vaihingen.sffnet import *

ablation = 'no_detail'
data_root = Path('/data2/tangyangpu/WorkSpace/RS/datasets/vaihingenR')
pretrained_ckpt_path = Path('/data2/tangyangpu/WorkSpace/RS/pretrained/round12/convnext_base.dinov3_lvd1689m/model.safetensors')
run_dir = root / 'experiments' / 'vaihingen_ablation' / ablation
weights_path = run_dir
checkpoint = run_dir / 'best_model.pth'
