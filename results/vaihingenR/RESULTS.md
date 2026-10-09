数值为百分比（Kappa 也乘100展示）。mIoU/mF1/mPrecision/mRecall/mAcc/mDice取前五类，排除Clutter；OA/FWIoU/Kappa使用全部六类；ignore_label=6。mAcc=宏平均Recall，Dice=F1。TTA=D4×[0.75,1,1.5]，24次logits平均，不保存预测图。沿用test作为验证集选择最佳权重，因此这些结果不是独立盲测。

## c10_dino_base — DINOv3 Base容量对照

最佳验证轮次：13；验证mIoU：85.8819；状态：completed

最佳权重：/data2/tangyangpu/WorkSpace/RS/checkpoint/round12/round12_vaihingen_105e_bs4_seed42/c10_dino_base/attempt_1/best_model.pth

| OA | mIoU | mF1 | mPrecision | mRecall | mAcc | mDice | FWIoU | Kappa | mIoU6 | mF1_6 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 94.1881 | 86.5963 | 92.6653 | 92.4036 | 92.9331 | 92.9331 | 92.6653 | 89.2874 | 91.6148 | 78.7883 | 86.7020 |

| 类别 | IoU | F1 | Precision | Recall | Dice | 像素数 |
|---|---:|---:|---:|---:|---:|---:|
| ImSurf | 94.6439 | 97.2483 | 97.2228 | 97.2737 | 97.2483 | 50960691 |
| Building | 93.8924 | 96.8500 | 96.8396 | 96.8604 | 96.8500 | 21949408 |
| LowVeg | 75.9144 | 86.3083 | 86.5228 | 86.0949 | 86.3083 | 17370785 |
| Tree | 83.6739 | 91.1114 | 90.0315 | 92.2175 | 91.1114 | 18512098 |
| Car | 84.8571 | 91.8083 | 91.4013 | 92.2190 | 91.8083 | 795433 |
| Clutter | 39.7484 | 56.8856 | 96.2548 | 40.3728 | 56.8856 | 692551 |

Checkpoint SHA256: 39a13d7eccf921cb8ef2f633d5eac01163adfc06e8ea32eb782c82c10f5deb20
测试图像：113；split：/data2/tangyangpu/WorkSpace/RS/datasets/vaihingenR/test；seed：42
