"""Evaluate a bundled complete segmentation checkpoint without backbone weights."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from geoseg.models.SFFNet.SFFNet import SFFNet
from geoseg.datasets.vaihingen_dataset import VaihingenDataset, val_aug
from tools.evaluation import evaluate_tta, sha256
from tools.reporting import write_json, render_model

from tools.cfg import py2cfg
ROOT = Path(__file__).resolve().parents[1]


def main(dataset_name):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config_path', type=Path, required=True)
    parser.add_argument('-o', '--output_path', type=Path)
    parser.add_argument('-t', '--tta', choices=['d4', 'glf'], default='d4',
                        help='d4: historical 24 views; glf: Vaihingen 24 views, Potsdam 8 views')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--no-save', action='store_true', help='Evaluate metrics without saving prediction masks')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--prediction-dir', type=Path)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error('Invalid shard index/count')
    config = py2cfg(args.config_path)
    if config.dataset != dataset_name:
        raise ValueError('Dataset entry and config disagree')
    args.dataset = config.dataset
    args.data_root = args.data_root or Path(config.data_root)
    args.output = args.output_path or ROOT/'outputs'/args.dataset
    args.workers = config.num_workers
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose one GPU with CUDA_VISIBLE_DEVICES')
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    entry = json.loads((ROOT / 'manifest.json').read_text())[args.dataset]
    checkpoint = args.checkpoint or Path(config.checkpoint)
    digest = sha256(checkpoint)
    if checkpoint.resolve() == (ROOT/entry['checkpoint']).resolve() and digest != entry['sha256']:
        raise ValueError('Checkpoint SHA256 mismatch')
    if digest != entry['sha256']:
        entry = dict(entry, best_epoch=None, best_val_miou=None)
    model = SFFNet(ablation=getattr(config, 'ablation', 'none'))
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.cuda().eval()
    split = args.data_root / 'test'
    dataset = VaihingenDataset(str(split), mode='test', transform=val_aug)
    images = {p.stem for p in (split / 'images_1024').glob('*.tif')}
    masks = {p.stem for p in (split / 'masks_1024').glob('*.png')}
    if not images or images != masks:
        raise ValueError('Empty or unpaired test data')
    full_ids = sorted(images)
    dataset.img_ids = full_ids[args.shard_index::args.num_shards]
    if not dataset.img_ids:
        raise ValueError("Empty shard")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    def progress(**values):
        write_json(args.output/'progress.json', dict(status='testing', total=len(dataset), **values))
    progress(tta_images=0)
    scales = None if args.tta == 'glf' and dataset_name == 'potsdamR' else (.75, 1., 1.5)
    result = evaluate_tta(model, loader, progress=progress, scales=scales,
                          prediction_dir=None if args.no_save else (args.prediction_dir or args.output))
    result.update(shard_index=args.shard_index, num_shards=args.num_shards, image_ids=dataset.img_ids,
                  tta_protocol=args.tta, batch_size=args.batch_size)
    result.update(checkpoint_sha256=digest, best_epoch=entry['best_epoch'], seed=42,
                  split=str(split.resolve()), best_checkpoint=str(checkpoint),
                  image_ids_sha256=__import__('hashlib').sha256('\n'.join(dataset.img_ids).encode()).hexdigest())
    row = dict(model='c10_dino_base', status='completed', best_epoch=entry['best_epoch'],
               best_val_miou=entry['best_val_miou'], best_checkpoint=str(checkpoint), tta=result)
    write_json(args.output / 'tta_metrics.json', result)
    render_model(row, args.output / 'RESULTS.md')
    write_json(args.output/'progress.json',dict(status='completed',tta_images=len(dataset),total=len(dataset)))
    print(json.dumps({k:result[k] for k in ['OA', 'mIoU', 'mF1', 'mPrecision', 'mRecall']}, indent=2))
