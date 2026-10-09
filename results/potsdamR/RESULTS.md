数值为百分比（Kappa 也乘100展示）。mIoU/mF1/mPrecision/mRecall/mAcc/mDice取前五类，排除Clutter；OA/FWIoU/Kappa使用全部六类；ignore_label=6。mAcc=宏平均Recall，Dice=F1。TTA=D4×[0.75,1,1.5]，24次logits平均，不保存预测图。沿用test作为验证集选择最佳权重，因此这些结果不是独立盲测。

## c10_dino_base — DINOv3 Base，Potsdam重新训练

最佳验证轮次：12；验证mIoU：87.8231；状态：completed

最佳权重：/data2/tangyangpu/WorkSpace/RS/checkpoint/potsdam_c10/potsdam_c10_105e_bs4_seed42/c10_dino_base/attempt_1/best_model.pth

| OA | mIoU | mF1 | mPrecision | mRecall | mAcc | mDice | FWIoU | Kappa | mIoU6 | mF1_6 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 92.4004 | 88.2496 | 93.6471 | 93.0264 | 94.2959 | 94.2959 | 93.6471 | 86.1055 | 89.8795 | 80.7209 | 88.0752 |

| 类别 | IoU | F1 | Precision | Recall | Dice | 像素数 |
|---|---:|---:|---:|---:|---:|---:|
| ImSurf | 89.9848 | 94.7284 | 93.2578 | 96.2462 | 94.7284 | 171963343 |
| Building | 94.8512 | 97.3576 | 97.2854 | 97.4299 | 97.3576 | 116645471 |
| LowVeg | 80.0962 | 88.9482 | 87.2412 | 90.7234 | 88.9482 | 96433432 |
| Tree | 82.2270 | 90.2468 | 91.0842 | 89.4246 | 90.2468 | 79936961 |
| Car | 94.0888 | 96.9544 | 96.2635 | 97.6552 | 96.9544 | 7445584 |
| Clutter | 43.0774 | 60.2155 | 81.2100 | 47.8463 | 60.2155 | 19920368 |

Checkpoint SHA256: 2a1f9f276c3eb6b39874217cab46b56bbe608694713de565562dbd7626f42e1b
测试图像：504；split：/data2/tangyangpu/WorkSpace/RS/datasets/potsdamR/test；seed：42
