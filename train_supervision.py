"""Original training entry and -c interface, using the established C10 training loop."""
import argparse
from pathlib import Path
from tools.cfg import py2cfg
from tools import training

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--dino-pretrained', type=Path)
    parser.add_argument('--smoke', action='store_true')
    return parser.parse_args()

def main():
    cli = get_args()
    config = py2cfg(cli.config_path)
    pretrained = cli.dino_pretrained or config.pretrained_ckpt_path
    if not pretrained:
        raise SystemExit('Set pretrained_ckpt_path in config or pass --dino-pretrained for fresh training.')
    args = training.parser().parse_args([
        '--model', 'c10_dino_base', '--ablation', getattr(config, 'ablation', 'none'), '--dataset', config.dataset,
        '--data-root', str(cli.data_root or config.data_root),
        '--dino-pretrained', str(pretrained), '--run-dir', str(config.run_dir),
        '--best-checkpoint', str(Path(config.weights_path)/(config.weights_name+'.pth')),
        '--epochs', str(config.max_epoch), '--batch-size', str(config.train_batch_size),
        '--val-batch-size', str(config.val_batch_size), '--workers', str(config.num_workers),
        '--seed', str(config.seed)] + (['--smoke'] if cli.smoke else []))
    training.run(args)

if __name__ == '__main__':
    main()
