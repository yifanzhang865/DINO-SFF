数值为百分比（Kappa 也乘100展示）。mIoU/mF1/mPrecision/mRecall/mAcc/mDice取前五类，排除Clutter；OA/FWIoU/Kappa使用全部六类；ignore_label=6。mAcc=宏平均Recall，Dice=F1。TTA=D4×[0.75,1,1.5]，24次logits平均，不保存预测图。沿用test作为验证集选择最佳权重，因此这些结果不是独立盲测。

## c10_dino_base — C10 DINOv3 Base

最佳验证轮次：27；验证mIoU：85.9421；状态：completed

最佳权重：/data2/tangyangpu/WorkSpace/RS/model/best/c10_dino_base/experiments/vaihingen_ablation/no_adapter/best_model.pth

| OA | mIoU | mF1 | mPrecision | mRecall | mAcc | mDice | FWIoU | Kappa | mIoU6 | mF1_6 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 94.0354 | 86.3001 | 92.5017 | 92.0619 | 92.9645 | 92.9645 | 92.5017 | 89.0631 | 91.4117 | 78.7464 | 86.7737 |

| 类别 | IoU | F1 | Precision | Recall | Dice | 像素数 |
|---|---:|---:|---:|---:|---:|---:|
| ImSurf | 94.5477 | 97.1974 | 97.6406 | 96.7582 | 97.1974 | 50960691 |
| Building | 93.0577 | 96.4040 | 96.3189 | 96.4893 | 96.4040 | 21949408 |
| LowVeg | 75.9600 | 86.3378 | 84.2942 | 88.4830 | 86.3378 | 17370785 |
| Tree | 83.5225 | 91.0215 | 91.1550 | 90.8885 | 91.0215 | 18512098 |
| Car | 84.4127 | 91.5476 | 90.9009 | 92.2035 | 91.5476 | 795433 |
| Clutter | 40.9781 | 58.1340 | 97.5847 | 41.3980 | 58.1340 | 692551 |

Checkpoint SHA256: 50779e54a0aea83c0ed096b73ca04109be7a964a6ba9276ade0deb79059edd4b
测试图像：113；split：/data2/tangyangpu/WorkSpace/RS/datasets/vaihingenR/test；seed：42
