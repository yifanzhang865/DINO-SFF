数值为百分比（Kappa 也乘100展示）。mIoU/mF1/mPrecision/mRecall/mAcc/mDice取前五类，排除Clutter；OA/FWIoU/Kappa使用全部六类；ignore_label=6。mAcc=宏平均Recall，Dice=F1。TTA=D4×[0.75,1,1.5]，24次logits平均，不保存预测图。沿用test作为验证集选择最佳权重，因此这些结果不是独立盲测。

## c10_dino_base — C10 DINOv3 Base

最佳验证轮次：27；验证mIoU：85.8376；状态：completed

最佳权重：/data2/tangyangpu/WorkSpace/RS/model/best/c10_dino_base/experiments/vaihingen_ablation/no_reverse/best_model.pth

| OA | mIoU | mF1 | mPrecision | mRecall | mAcc | mDice | FWIoU | Kappa | mIoU6 | mF1_6 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 94.0701 | 86.4360 | 92.5717 | 92.0832 | 93.0740 | 93.0740 | 92.5717 | 89.1021 | 91.4479 | 78.8629 | 86.8354 |

| 类别 | IoU | F1 | Precision | Recall | Dice | 像素数 |
|---|---:|---:|---:|---:|---:|---:|
| ImSurf | 94.5054 | 97.1751 | 97.2200 | 97.1303 | 97.1751 | 50960691 |
| Building | 93.6889 | 96.7416 | 96.7752 | 96.7081 | 96.7416 | 21949408 |
| LowVeg | 75.6797 | 86.1565 | 85.2410 | 87.0919 | 86.1565 | 17370785 |
| Tree | 83.3623 | 90.9263 | 90.7907 | 91.0624 | 90.9263 | 18512098 |
| Car | 84.9435 | 91.8589 | 90.3891 | 93.3773 | 91.8589 | 795433 |
| Clutter | 40.9978 | 58.1538 | 97.1285 | 41.5008 | 58.1538 | 692551 |

Checkpoint SHA256: 85d3d293495f825d7d415b011f192201cf02bbdc719a205a4c83b79bbfae3c55
测试图像：113；split：/data2/tangyangpu/WorkSpace/RS/datasets/vaihingenR/test；seed：42
