数值为百分比（Kappa 也乘100展示）。mIoU/mF1/mPrecision/mRecall/mAcc/mDice取前五类，排除Clutter；OA/FWIoU/Kappa使用全部六类；ignore_label=6。mAcc=宏平均Recall，Dice=F1。TTA=D4×[0.75,1,1.5]，24次logits平均，不保存预测图。沿用test作为验证集选择最佳权重，因此这些结果不是独立盲测。

## c10_dino_base — C10 DINOv3 Base

最佳验证轮次：38；验证mIoU：86.0216；状态：completed

最佳权重：/data2/tangyangpu/WorkSpace/RS/model/best/c10_dino_base/experiments/vaihingen_ablation/no_scale_route/best_model.pth

| OA | mIoU | mF1 | mPrecision | mRecall | mAcc | mDice | FWIoU | Kappa | mIoU6 | mF1_6 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 94.1176 | 86.5210 | 92.6156 | 92.3205 | 92.9208 | 92.9208 | 92.6156 | 89.1812 | 91.5121 | 79.5910 | 87.5151 |

| 类别 | IoU | F1 | Precision | Recall | Dice | 像素数 |
|---|---:|---:|---:|---:|---:|---:|
| ImSurf | 94.5512 | 97.1993 | 97.0982 | 97.3005 | 97.1993 | 50960691 |
| Building | 93.9704 | 96.8915 | 97.3146 | 96.4720 | 96.8915 | 21949408 |
| LowVeg | 75.4913 | 86.0342 | 86.2260 | 85.8433 | 86.0342 | 17370785 |
| Tree | 83.3925 | 90.9443 | 89.7359 | 92.1856 | 90.9443 | 18512098 |
| Car | 85.1998 | 92.0085 | 91.2280 | 92.8025 | 92.0085 | 795433 |
| Clutter | 44.9412 | 62.0130 | 95.2191 | 45.9787 | 62.0130 | 692551 |

Checkpoint SHA256: c0f7694992790016816e43ede1d7af2bc26199a28cd8d9efff7db3fb670c557a
测试图像：113；split：/data2/tangyangpu/WorkSpace/RS/datasets/vaihingenR/test；seed：42
