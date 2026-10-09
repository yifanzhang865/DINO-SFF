# DINO-SFF

DINO-SFF is a remote-sensing semantic segmentation model with a DINOv3 ConvNeXt Base backbone and a multi-scale fusion decoder. The implementation retains the historical internal name `c10_dino_base` for checkpoint compatibility.

## Evaluation results

The following results use the dataset-specific TTA settings in the local GLF implementation. All numbers are percentages; macro metrics exclude Clutter. The test split was also used for checkpoint selection, so these are **not independent held-out test estimates**.

| Dataset | Images | TTA | OA | mIoU | mF1 |
|---|---:|---|---:|---:|---:|
| Vaihingen | 113 | D4 × [0.75, 1.0, 1.5], 24 views | 94.1881 | 86.5962 | 92.6652 |
| Potsdam | 504 | single-scale D4, 8 views | 92.3699 | 88.1914 | 93.6117 |

- [Full evaluation and per-class metrics](experiments/glf_tta/RESULTS.md)
- [Six single-component ablations on Vaihingen](experiments/vaihingen_ablation/COMPARISON.md)
- [FLOPs, memory and latency measurements](experiments/model_profile/PROFILE.md): 89.44M parameters; 754.02 GFLOPs at 1024×1024, counting one multiply-add as two FLOPs.

## Quick start

Use Python 3.10 and a CUDA-compatible PyTorch environment matching [requirements-rs.txt](requirements-rs.txt). Dataset layout: `train/images_1024/*.tif`, `train/masks_1024/*.png`, `test/images_1024/*.tif`, `test/masks_1024/*.png`.

Model weights and datasets are not stored in Git. Place the complete best checkpoints at `checkpoints/c10_dino_base_vaihingenR_best.pth` and `checkpoints/c10_dino_base_potsdamR_best.pth`; verify their SHA256 against [manifest.json](manifest.json). Training additionally requires the external DINOv3 Base initialization specified in [pretrained_manifest.json](pretrained_manifest.json).

```bash
CUDA_VISIBLE_DEVICES=0 python vaihingen_test.py -c config/vaihingen/sffnet.py --data-root /path/to/vaihingenR -t glf --batch-size 2 --no-save -o outputs/vaihingen_glf
CUDA_VISIBLE_DEVICES=0 python potsdam_test.py -c config/potsdam/sffnet.py --data-root /path/to/potsdamR -t glf --batch-size 2 --no-save -o outputs/potsdam_glf
CUDA_VISIBLE_DEVICES=0 python train_supervision.py -c config/vaihingen/sffnet.py --data-root /path/to/vaihingenR --dino-pretrained /path/to/dinov3_base/model.safetensors
```

Experiment orchestration scripts retain original machine paths for provenance; pass dataset paths through the main training/testing entrypoints or adapt those scripts to your environment. No dataset is included. Source attribution and upstream terms are in [NOTICE.md](NOTICE.md).

## Citation

Repository: https://github.com/yifanzhang865/DINO-SFF

Fixed code version: [v1.0.0](https://github.com/yifanzhang865/DINO-SFF/tree/v1.0.0).

Cite the exact code version used in your work. GitHub's citation metadata is provided in [CITATION.cff](CITATION.cff); it describes this software repository, not an associated published paper.

```bibtex
@software{dino_sff_2026,
  author = {yifanzhang865},
  title = {{DINO-SFF}: DINOv3-based Remote Sensing Semantic Segmentation},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/yifanzhang865/DINO-SFF}
}
```

## Implementation and original experiment documentation

按GLF-Net原目录组织，保留train_supervision.py、vaihingen_test.py和potsdam_test.py入口及-c配置参数。
C10所有网络实现统一在geoseg/models/SFFNet/SFFNet.py：DINOv3 Base构建、Restoration适配器、动态采样、尺度路由、原型融合、反向细化及辅助头。
不再保留models/round8.py、w02或u05等实验文件，也不再依赖历轮实验目录。

```text
DINO-SFF/
├── train_supervision.py
├── vaihingen_test.py
├── potsdam_test.py
├── config/
│   ├── vaihingen/sffnet.py
│   └── potsdam/sffnet.py
├── geoseg/
│   ├── models/SFFNet/SFFNet.py   # 唯一模型实现文件
│   ├── datasets/               # 数据加载与增强
│   └── losses/segmentation.py  # C10损失
├── tools/                     # 配置、训练流程、评估及报告
├── checkpoints/               # 两份完整最佳模型权重
├── configs/                   # 原始运行配置存档
└── results/                   # 两个数据集完整指标存档
```

保留的是原版工程结构、模型导入路径及训练/测试命令入口。内部训练仍采用已得到这两份权重的C10训练流程与多头损失，未切换为原始SFFNet的Lightning训练逻辑。
网络参数名与计算保持一致；两个checkpoint严格加载通过，整理前后全部输出逐元素完全相同，训练反向梯度有限。

## 环境与数据

Python3.10，依赖见requirements-rs.txt。保留适配机器CUDA的PyTorch构建；Catalyst22.4安装需兼容其元数据的pip版本（<24.1）。
编辑config/<dataset>/sffnet.py的data_root，或用--data-root覆盖。
数据结构为train/test各含images_1024/*.tif和masks_1024/*.png。
Potsdam仅排除已确认损坏的训练图top_potsdam_7_8_2_34，原图不修改，完整504张测试图均保留。
不包含数据集或单独的backbone预训练文件。完整模型权重包含训练后的backbone参数，测试无需下载或额外预训练文件。

## 训练入口

配置中的pretrained_ckpt_path填写外部DINOv3 Base safetensors路径；也可使用--dino-pretrained覆盖。
固定版本与SHA256见pretrained_manifest.json。

```bash
CUDA_VISIBLE_DEVICES=0 python train_supervision.py -c config/vaihingen/sffnet.py --data-root /path/to/vaihingenR --dino-pretrained /path/to/dinov3_base/model.safetensors
CUDA_VISIBLE_DEVICES=0 python train_supervision.py -c config/potsdam/sffnet.py --data-root /path/to/potsdamR --dino-pretrained /path/to/dinov3_base/model.safetensors
```

默认batch4、105epochs、seed42、FP32，512裁剪、mosaic=.25，验证batch2；Lookahead AdamW主干6e-5/其他6e-4，weight_decay=.01，梯度裁剪5，余弦热重启15/2。
保存验证mIoU最高的完整state_dict，训练结束自动加载最佳权重执行TTA。配置run_dir和weights_path可修改输出位置。

## 测试入口

```bash
CUDA_VISIBLE_DEVICES=0 python vaihingen_test.py -c config/vaihingen/sffnet.py --data-root /path/to/vaihingenR -t d4 -o outputs/vaihingenR --no-save
CUDA_VISIBLE_DEVICES=0 python potsdam_test.py -c config/potsdam/sffnet.py --data-root /path/to/potsdamR -t d4 -o outputs/potsdamR --no-save
```

GPU0仅为命令示例，请选择空闲GPU。默认加载checkpoints内对应数据集最佳权重，-o指定指标输出目录。
-t d4沿用本实验24路多尺度D4 TTA；默认保存带类别调色板的PNG预测图；--no-save可禁用图像输出。
配置checkpoint或--checkpoint可以指定另一份完整模型权重；非打包权重不会标注打包模型的最佳轮次。
支持的入口参数以--help为准，不包含历史实验的--round6-model等选项。

另支持 `-t glf --batch-size 2`，匹配本地 GLF-Net-main 的 `d4` 测试设置：
Vaihingen 为 D4×[0.75,1,1.5] 共24路，Potsdam 为不加缩放的D4共8路；均采用logits平均。
`-t d4` 保持两个数据集原有的24路行为。GLF口径实测及逐类指标见 `experiments/glf_tta/RESULTS.md`。

## 已完成的TTA结果（%）

| 数据集 | 最佳轮 | OA | mIoU | mF1 |
|---|---:|---:|---:|---:|
| VaihingenR | 13 | 94.1881 | 86.5963 | 92.6653 |
| PotsdamR | 12 | 92.4004 | 88.2496 | 93.6471 |

详细结果见results/vaihingenR/RESULTS.md和results/potsdamR/RESULTS.md。
总体包含OA/mIoU/mF1/mPrecision/mRecall/mAcc/mDice/FWIoU/Kappa及六类均值；每类含IoU/F1/Precision/Recall/Dice/像素数。
主宏平均使用前五类，排除Clutter；OA/FWIoU/Kappa采用全部六类；ignore=6；mAcc=平均Recall，Dice=F1。
沿用test作验证及最佳权重选择，结果不是独立盲测。TTA是D4×[0.75,1,1.5]共24路logits平均。

## 验证与追溯

manifest.json包含两份权重相对路径、SHA256、原始路径、最佳轮和TTA结果。
历史记录中的绝对路径仅作追溯，运行不依赖原实验目录。
verification.json记录两份权重严格加载、全部输出与原实现逐元素相同、有限梯度验证。
entrypoint_verification.json记录真实batch4训练、最佳权重保存重载及两个数据集测试入口的24路TTA检查。
检查用少量样本结果不替代results/中的完整数据集结果。

本次保存图像的完整TTA结果位于tta_d4_results/vaihigen和tta_d4_results/postdam。PNG调色板：不透水面白、建筑蓝、低矮植被青、树绿、车黄、杂类红；PNG像素值仍为类别0–5。各子目录含RESULTS.md和tta_metrics.json。
