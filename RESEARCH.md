# 第六轮：DINOv3 主干迁移与第五轮有效模块融合

## 1. 第五轮结果驱动的选择

前五轮 50 个模型均完成 105 epochs；最佳值已和逐 epoch CSV 核对，详见 [HISTORY.md](HISTORY.md)。

| 第五轮模型 | 最佳 mIoU (%) | 末十轮均值 (%) |
|---|---:|---:|
| u05_restoration_attention | 84.9244 | 84.7612 |
| u08_convex_refinement | 84.9097 | 84.7381 |
| u06_multiplicative_gates | 84.8974 | 84.7152 |
| u10_hybrid_readout | 84.8578 | 84.5251 |
| u04_spectral_operator | 84.8318 | 84.7000 |
| u03_large_kernel | 84.8204 | 84.6653 |
| u01_dense_grn | 84.8111 | 84.3100 |
| u09_metric_embedding | 84.7330 | 84.6344 |
| u07_adaptive_heat | 84.7188 | 84.6129 |
| u02_antialias_detail | 84.5493 | 84.3457 |

u05 相对此前历史最佳 84.8439% 提升约 0.0804 个百分点，但末十轮均值仅提升约 0.0072 个百分点。说明峰值改善不能替代稳定性和多种子证据。本轮主要继承注意力、乘性门控、凸组合三类正向设计，不继续叠加抗混叠、各向异性传导或度量损失等未显示优势的修改。

## 2. 为什么试 DINOv3 ConvNeXt-Tiny

[DINOv3](https://arxiv.org/abs/2508.10104) 研究通用自监督视觉特征；[Meta 模型卡](https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m) 将 ConvNeXt 系列列为从 ViT-7B 蒸馏的模型。Tiny 是 LVD-1689M 网络图像预训练，而非 SAT-493M 遥感预训练，不能混淆两者。

Tiny 的主干约 27.8M 参数，四阶段输出 96/192/384/768 通道、步长 4/8/16/32，能连接当前多尺度解码器。更丰富的预训练可能弥补较小容量，但这只是迁移假设，不预先承诺优于 vHeat。

本次 Meta 原始 Hugging Face 仓库返回 401，未使用该受限入口。改用 **timm 维护者公开发布的发行版** [timm/convnext_tiny.dinov3_lvd1689m](https://huggingface.co/timm/convnext_tiny.dinov3_lvd1689m)，该仓库 API 显示 `gated=False`，当前安装的 timm 1.0.29 原生支持这个模型名。不是不明来源镜像，也不是 ImageNet 普通 ConvNeXt 权重。

- 固定 revision：`39c7bb32f83c11bbb13cb23058f8852a4dac88ed`。
- 文件大小：`111298008` bytes。
- SHA256：`a7637753a2e85d709d1a5474e13382d39e7bc581a35868b477585c970f36847f`。
- 权重路径：`/data2/tangyangpu/WorkSpace/RS/pretrained/dinov3_convnext_tiny/model.safetensors`。
- 适用许可为 DINOv3 License，不能将 timm 的代码许可当作权重许可。使用/分发时保留下载目录的 `LICENSE.md` 与来源信息。

## 3. 主干接入的实现约束

`models/dino_backbone.py` 校验整文件 SHA256，然后以 `strict=True` 加载全部 checkpoint 张量。加载后移除池化 head 外壳，将预训练的最终通道 LayerNorm 保留在最终稠密特征上，避免无梯度的全局 head 参数。所有四阶段预训练参数都保留。

本轮 DINOv3 路径不构建 vHeat，不将 vHeat 权重塞进 DINO 网络；vHeat 两个版本仍使用原 `vHeat_base.pth`。训练日志写入实际 backbone、实际权重路径、总参数与可训练参数，checkpoint 不完整时直接失败，禁止静默随机初始化。

输入沿用已有 ImageNet mean/std，与 timm 模型配置一致；不套用分类中心裁剪，以免改变分割图像/标签几何。DINO 原模型 drop-path=0，vHeat 保留原 0.5，属于主干固有差异；因此不是严格的仅一个结构因素消融。

## 4. 十个版本

| 版本 | 主干 | 改动/检验目标 |
|---|---|---|
| W01 w01_dino_reference | DINOv3 Tiny | 主干替换及四尺度通道适配，保留 S09 解码器。刻意保留朴素迁移对照，避免无法判断新预训练本身的作用 |
| W02 w02_dino_restoration | DINOv3 Tiny | 四阶段内插入 U05 式通道转置注意力、局部 qkv 卷积与受限残差 |
| W03 w03_dino_multiplicative | DINOv3 Tiny | 四阶段乘性双支门控、全局通道调制、GRN 残差适配 |
| W04 w04_dino_attention_convex | DINOv3 Tiny | W02 主干与 U08 两步边缘约束凸组合联合使用 |
| W05 w05_dino_hierarchical | DINOv3 Tiny | 浅层乘性纹理、深层通道上下文；高分辨率跨尺度门控融合 |
| W06 w06_dino_frozen_adapters | DINOv3 Tiny | 冻结原主干和最后归一化；训练四阶段注意力适配器、解码器和凸细化，检验保留基础模型参数是否更稳 |
| W07 w07_dino_multiscale_convex | DINOv3 Tiny | 深层注意力、三层自上而下尺度门控、边缘约束凸细化 |
| W08 w08_dino_image_detail | DINOv3 Tiny | 增加重叠卷积图像细节支路，弥补 stride-4 patch stem；深层注意力与凸细化 |
| W09 w09_vheat_attention_convex | vHeat Base | U05 注意力主干与 U08 凸细化融合；作为 W04 的另一预训练主干对照 |
| W10 w10_vheat_hierarchical_convex | vHeat Base | 浅层乘性/深层注意力分工、跨尺度细节、凸细化融合 |

适配器插入编码阶段，输出继续流向后续 stage，而非只修饰解码器跳接。W06 冻结的是参数，不是中间激活；不能用整个 encoder 的 `no_grad()` 包裹，否则浅层适配器无法通过后续冻结层取得梯度。

## 5. 论文与既有实验证据

- [Restormer](https://arxiv.org/abs/2111.09881)：通道转置注意力；在本项目第五轮 U05 中已显示正向峰值结果，不等于证明在 DINO 上同样有效。
- [NAFNet](https://arxiv.org/abs/2204.04676)：乘性门控启发 U06，本轮迁移其简化模块，不复制完整复原网络。
- [RAFT](https://arxiv.org/abs/2003.12039)：邻域凸权重的设计启发 U08。本轮是同分辨率 logits 修正，不是完整光流模型。
- DINOv3 模型卡建议先考虑冻结特征，故加入 W06；其余版本维持全参数微调。W06 仍有可训练的阶段适配器，因此不是完全固定特征的线性探测。

## 6. 风险与比较原则

八个 DINO 版本改变了架构、预训练数据和参数量，可能改善语义泛化，也可能因为 RGB/遥感域差异、容量变小或全量微调破坏表征而降低精度。保留 W01/W06 和两种 vHeat 融合对照来定位原因。W08 专门检查小目标细节，但也增加显存和过拟合风险。

固定 train 741 / val 113、512 训练、1024 验证、batch2、seed42、105 epochs、FP32、AdamW+Lookahead、backbone LR 6e-5 / decoder 6e-4、weight decay 0.01、余弦重启、梯度裁剪 5。不保存分割/优化器权重；新下载的**初始化主干权重**保留用于所有模型重现。

无 D4/MS-D4；主指标五类 mIoU，另存六类和逐类 IoU。先验证权重、梯度、冻结参数与真实尺寸运行，再开始正式比较。每轮最高值、末十轮均值和开销都应报告，不用小样本 smoke 代替真实结果。反复使用 `test` 目录做选型已有选择偏差，最终研究结论需要独立测试、多种子与进一步消融。
