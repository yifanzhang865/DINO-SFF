数值为百分比（Kappa 也乘100展示）。mIoU/mF1/mPrecision/mRecall/mAcc/mDice取前五类，排除Clutter；OA/FWIoU/Kappa使用全部六类；ignore_label=6。mAcc=宏平均Recall，Dice=F1。TTA=D4×[0.75,1,1.5]，24次logits平均，不保存预测图。沿用test作为验证集选择最佳权重，因此这些结果不是独立盲测。

## c10_dino_base — C10 DINOv3 Base

最佳验证轮次：13；验证mIoU：85.7247；状态：completed

最佳权重：/data2/tangyangpu/WorkSpace/RS/model/best/c10_dino_base/experiments/vaihingen_ablation/no_detail/best_model.pth

| OA | mIoU | mF1 | mPrecision | mRecall | mAcc | mDice | FWIoU | Kappa | mIoU6 | mF1_6 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 94.2083 | 86.4171 | 92.5653 | 92.2415 | 92.8927 | 92.8927 | 92.5653 | 89.3218 | 91.6461 | 79.3290 | 87.3050 |

| 类别 | IoU | F1 | Precision | Recall | Dice | 像素数 |
|---|---:|---:|---:|---:|---:|---:|
| ImSurf | 94.6616 | 97.2576 | 97.2825 | 97.2327 | 97.2576 | 50960691 |
| Building | 93.6388 | 96.7149 | 96.4730 | 96.9581 | 96.7149 | 21949408 |
| LowVeg | 76.1604 | 86.4671 | 86.1723 | 86.7640 | 86.4671 | 17370785 |
| Tree | 83.7891 | 91.1796 | 90.8089 | 91.5533 | 91.1796 | 18512098 |
| Car | 83.8355 | 91.2071 | 90.4710 | 91.9552 | 91.2071 | 795433 |
| Clutter | 43.8884 | 61.0034 | 95.6612 | 44.7798 | 61.0034 | 692551 |

Checkpoint SHA256: 8ee7e8bac2558a1ec58306b2ea5108068ace3819d1d7a4e47fc66827d2f538d8
测试图像：113；split：/data2/tangyangpu/WorkSpace/RS/datasets/vaihingenR/test；seed：42
