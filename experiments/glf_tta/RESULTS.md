# C10 完整模型：与本地 GLF 一致的 TTA 指标

完成时间：2026-10-09T03:30:27+00:00

使用两个数据集原有最佳权重，不重新训练。Vaihingen 为 24 路多尺度 D4；Potsdam 为 8 路单尺度 D4。FP32、batch=2；分片测试后累加混淆矩阵，未平均分片指标。

mIoU/mF1/mPrecision/mRecall/mDice/mAcc 为前五类均值，排除 Clutter；OA/FWIoU/Kappa 为六类。指标统一以百分比显示（Kappa 同样乘100）。沿用 test 集选最佳权重，不是独立盲测。

| 数据集 | 图数 | TTA | OA | mIoU | mF1 | mPrecision | mRecall | mDice | mAcc | FWIoU | Kappa | mIoU6 | mF1_6 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vaihingenR | 113 | 24 路 | 94.1881 | 86.5962 | 92.6652 | 92.4035 | 92.9331 | 92.6652 | 92.9331 | 89.2874 | 91.6147 | 78.7885 | 86.7021 |
| potsdamR | 504 | 8 路 | 92.3699 | 88.1914 | 93.6117 | 93.0032 | 94.2478 | 93.6117 | 94.2478 | 86.0757 | 89.8420 | 80.6591 | 88.0326 |

## 与原先统一 24 路 TTA 的差值（百分点）

| 数据集 | ΔOA | ΔmIoU | ΔmF1 |
|---|---:|---:|---:|
| vaihingenR | -0.0000 | -0.0001 | -0.0001 |
| potsdamR | -0.0305 | -0.0582 | -0.0354 |

## vaihingenR 逐类指标

| 类别 | IoU | F1/Dice | Precision | Recall | 像素数 |
|---|---:|---:|---:|---:|---:|
| ImSurf | 94.6439 | 97.2483 | 97.2228 | 97.2738 | 50960691 |
| Building | 93.8924 | 96.8500 | 96.8398 | 96.8602 | 21949408 |
| LowVeg | 75.9143 | 86.3083 | 86.5224 | 86.0953 | 17370785 |
| Tree | 83.6739 | 91.1114 | 90.0318 | 92.2171 | 18512098 |
| Car | 84.8567 | 91.8081 | 91.4007 | 92.2191 | 795433 |
| Clutter | 39.7495 | 56.8868 | 96.2549 | 40.3739 | 692551 |

GLF 源文件：`/data2/tangyangpu/WorkSpace/RS/model/GLF-Net-main/GLF-Net-main/vaihingen_test.py`；SHA256：`87b5e98a7ca9335c45b6774fc37acb20902a01fbbb2e3689da0f3af79ded02e7`。
权重 SHA256：`39a13d7eccf921cb8ef2f633d5eac01163adfc06e8ea32eb782c82c10f5deb20`。


## potsdamR 逐类指标

| 类别 | IoU | F1/Dice | Precision | Recall | 像素数 |
|---|---:|---:|---:|---:|---:|
| ImSurf | 90.0164 | 94.7460 | 93.4128 | 96.1177 | 171963343 |
| Building | 94.9191 | 97.3933 | 97.2448 | 97.5422 | 116645471 |
| LowVeg | 79.9150 | 88.8364 | 87.0686 | 90.6774 | 96433432 |
| Tree | 82.1247 | 90.1851 | 91.0592 | 89.3276 | 79936961 |
| Car | 93.9820 | 96.8976 | 96.2304 | 97.5742 | 7445584 |
| Clutter | 42.9973 | 60.1372 | 79.9684 | 48.1874 | 19920368 |

GLF 源文件：`/data2/tangyangpu/WorkSpace/RS/model/GLF-Net-main/GLF-Net-main/potsdam_test.py`；SHA256：`f5422457f4e6bc8407a1ead82fe45c99e708e203c11feb49bd80e444e105b1a9`。
权重 SHA256：`2a1f9f276c3eb6b39874217cab46b56bbe608694713de565562dbd7626f42e1b`。

## 复现

```bash
CUDA_VISIBLE_DEVICES=3 /data2/tangyangpu/RS/bin/python vaihingen_test.py -c config/vaihingen/sffnet.py --data-root /data2/tangyangpu/WorkSpace/RS/datasets/vaihingenR -t glf --batch-size 2 --no-save -o outputs/glf_vaihingen
CUDA_VISIBLE_DEVICES=4 /data2/tangyangpu/RS/bin/python potsdam_test.py -c config/potsdam/sffnet.py --data-root /data2/tangyangpu/WorkSpace/RS/datasets/potsdamR -t glf --batch-size 2 --no-save -o outputs/glf_potsdam
```
