数值为百分比（Kappa 也乘100展示）。mIoU/mF1/mPrecision/mRecall/mAcc/mDice取前五类，排除Clutter；OA/FWIoU/Kappa使用全部六类；ignore_label=6。mAcc=宏平均Recall，Dice=F1。TTA=D4×[0.75,1,1.5]，24次logits平均，不保存预测图。沿用test作为验证集选择最佳权重，因此这些结果不是独立盲测。

## c10_dino_base — C10 DINOv3 Base

最佳验证轮次：12；验证mIoU：85.7747；状态：completed

最佳权重：/data2/tangyangpu/WorkSpace/RS/model/best/c10_dino_base/experiments/vaihingen_ablation/no_exchange/best_model.pth

| OA | mIoU | mF1 | mPrecision | mRecall | mAcc | mDice | FWIoU | Kappa | mIoU6 | mF1_6 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 94.1814 | 86.5079 | 92.6153 | 92.1629 | 93.0883 | 93.0883 | 92.6153 | 89.2727 | 91.6110 | 78.9656 | 86.9146 |

| 类别 | IoU | F1 | Precision | Recall | Dice | 像素数 |
|---|---:|---:|---:|---:|---:|---:|
| ImSurf | 94.8359 | 97.3495 | 97.5128 | 97.1868 | 97.3495 | 50960691 |
| Building | 93.4246 | 96.6005 | 96.1842 | 97.0205 | 96.6005 | 21949408 |
| LowVeg | 75.8491 | 86.2661 | 87.0082 | 85.5366 | 86.2661 | 17370785 |
| Tree | 83.6192 | 91.0789 | 89.5568 | 92.6536 | 91.0789 | 18512098 |
| Car | 84.8107 | 91.7811 | 90.5523 | 93.0438 | 91.7811 | 795433 |
| Clutter | 41.2542 | 58.4113 | 95.7602 | 42.0217 | 58.4113 | 692551 |

Checkpoint SHA256: a1c19c8144057c132a23e84b96b97cdb10f70cd3a36623f72c3b63a747cb6079
测试图像：113；split：/data2/tangyangpu/WorkSpace/RS/datasets/vaihingenR/test；seed：42
