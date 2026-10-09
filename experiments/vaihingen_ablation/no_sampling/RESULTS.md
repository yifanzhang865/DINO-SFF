数值为百分比（Kappa 也乘100展示）。mIoU/mF1/mPrecision/mRecall/mAcc/mDice取前五类，排除Clutter；OA/FWIoU/Kappa使用全部六类；ignore_label=6。mAcc=宏平均Recall，Dice=F1。TTA=D4×[0.75,1,1.5]，24次logits平均，不保存预测图。沿用test作为验证集选择最佳权重，因此这些结果不是独立盲测。

## c10_dino_base — C10 DINOv3 Base

最佳验证轮次：15；验证mIoU：85.5832；状态：completed

最佳权重：/data2/tangyangpu/WorkSpace/RS/model/best/c10_dino_base/experiments/vaihingen_ablation/no_sampling/best_model.pth

| OA | mIoU | mF1 | mPrecision | mRecall | mAcc | mDice | FWIoU | Kappa | mIoU6 | mF1_6 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 94.1593 | 86.2072 | 92.4334 | 92.3047 | 92.5838 | 92.5838 | 92.4334 | 89.2344 | 91.5734 | 79.6013 | 87.6192 |

| 类别 | IoU | F1 | Precision | Recall | Dice | 像素数 |
|---|---:|---:|---:|---:|---:|---:|
| ImSurf | 94.7425 | 97.3003 | 97.2577 | 97.3429 | 97.3003 | 50960691 |
| Building | 93.6117 | 96.7005 | 96.6057 | 96.7954 | 96.7005 | 21949408 |
| LowVeg | 75.5415 | 86.0668 | 87.3577 | 84.8135 | 86.0668 | 17370785 |
| Tree | 83.5693 | 91.0493 | 89.2961 | 92.8728 | 91.0493 | 18512098 |
| Car | 83.5709 | 91.0503 | 91.0064 | 91.0942 | 91.0503 | 795433 |
| Clutter | 46.5717 | 63.5480 | 91.4976 | 48.6783 | 63.5480 | 692551 |

Checkpoint SHA256: 9f5f2f8ae055810390591a842b150b553b3bf65c1961e878b06a3110f4dd4cfc
测试图像：113；split：/data2/tangyangpu/WorkSpace/RS/datasets/vaihingenR/test；seed：42
