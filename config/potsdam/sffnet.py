from pathlib import Path
from geoseg.models.SFFNet.SFFNet import SFFNet

root = Path(__file__).resolve().parents[2]
dataset = "potsdamR"
data_root = root / "data" / dataset
pretrained_ckpt_path = None  # Set an external DINOv3 Base safetensors path for fresh training.
max_epoch = 105
train_batch_size = 4
val_batch_size = 2
num_workers = 4
seed = 42
num_classes = 6
ignore_index = 6
run_dir = root / "outputs" / "train_potsdamR"
weights_path = run_dir
weights_name = "best_model"
checkpoint = root / "checkpoints" / "c10_dino_base_potsdamR_best.pth"
monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = False
tta_scales = [0.75, 1.0, 1.5]
