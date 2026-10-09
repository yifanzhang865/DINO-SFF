# C10 DINOv3 Base FLOPs 与显存实测

测量时间：2026-10-09T02:48:41+00:00；GPU：NVIDIA GeForce RTX 3090；PyTorch：2.1.2+cu118。

参数量：**89,437,565（89.4376 M）**；FP32 参数内存：**341.18 MiB**；buffer：0.0000 MiB；权重文件：341.35 MiB。

| 场景 | Batch | 输入尺寸 | GMACs | GFLOPs | 峰值 allocated (MiB) | 峰值 reserved (MiB) | 中位耗时 (ms/次) |
|---|---:|---|---:|---:|---:|---:|---:|
| infer_512 | 1 | 512×512 | 94.253 | 188.506 | 522.40 | 596.00 | 28.98 |
| infer_1024 | 1 | 1024×1024 | 377.011 | 754.022 | 848.15 | 1004.00 | 78.54 |
| tta_1024 | 1 | 1024×1024 | 11498.834 | 22997.669 | 1319.92 | 1928.00 | 2295.29 |
| train_512_bs4 | 4 | 512×512 | 未测 | 未测 | 9114.17 | 9664.00 | 414.03 |

## 测量口径

- 完整模型、原版 Vaihingen 最佳权重，FP32，无 AMP。输入为固定 seed=42 的合成张量；训练标签包含 0–6。没有写回或修改最佳权重。
- FLOPs 使用 PyTorch 自带 FlopCounterMode 实际执行统计，乘法与加法各算一次，即 1 MAC=2 FLOPs；1G=10^9。覆盖卷积和矩阵乘法（包括注意力的 bmm/mm/addmm）。
- **这是卷积/矩阵乘法口径，并非全部算子的精确总运算数**：不计 bias 加法、归一化、激活、softmax、插值、grid_sample、池化、逐元素运算及 TTA 变换/融合。未计入算子的调用清单保存在 JSON。推理仍实际执行所有辅助头，未做部署裁剪。
- TTA 为 D4×[0.75,1,1.5] 共 24 路，FLOPs 对实际完整 TTA 路径计数；显存也是实际 wrapper 测量。TTA 顺序执行，不是 24 张同时送入模型。
- memory 指 GPU 显存。allocated 峰值为 PyTorch 张量分配峰值，包含模型、输入及中间结果；reserved 为缓存分配器预留峰值，包含可复用缓存。两者均不是仅增量激活内存，也不等于 nvidia-smi 的进程显存；CUDA context、外部库分配不保证包含。
- 每个场景在独立进程测量：预热 2 次，清空空闲缓存、重置峰值后测 5 次；计时经 CUDA synchronize，不包含数据读取和 CPU→GPU 传输。FLOPs 统计在显存/耗时测量之后单独执行。
- 训练使用原版多头损失、Lookahead AdamW、梯度裁剪 5，batch=4；峰值包含优化器状态及前向/反向/更新，5 次实测覆盖 Lookahead 的一个完整周期。训练 FLOPs 未测，不能直接按推理 FLOPs 相乘代替。
- cuDNN TF32=True，matmul TF32=False；cudnn benchmark=False，deterministic=True。

## 重现

```bash
CUDA_VISIBLE_DEVICES=4 MPLCONFIGDIR=/tmp/c10_profile_mpl /data2/tangyangpu/RS/bin/python -m tools.profile_model
```

权重 SHA256：`39a13d7eccf921cb8ef2f633d5eac01163adfc06e8ea32eb782c82c10f5deb20`。各场景详细结果见同目录 JSON。
