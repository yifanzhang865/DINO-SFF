# 文献、预训练与本轮适配

这些实现为论文思路的适配和自主组合，不宣称复现完整论文网络，也不预设精度提升。

| 来源 | 本轮用途 | 与原方法的关系 |
|---|---|---|
| [DINOv3 官方项目](https://github.com/facebookresearch/dinov3)、[Small 权重](https://huggingface.co/timm/convnext_small.dinov3_lvd1689m)、[Base 权重](https://huggingface.co/timm/convnext_base.dinov3_lvd1689m) | C01–09使用ConvNeXt Small；C10使用ConvNeXt Base | 使用对应公开预训练张量，strict加载342个张量；不使用历史分割权重 |
| [EMCAD，CVPR 2024](https://github.com/SLDGroup/EMCAD) | C01通道/空间门控与3/5/7深度卷积 | 迁移轻量多尺度细化思路，残差初值0.01；不是完整EMCAD解码器 |
| [FADC，CVPR 2024](https://github.com/Linwei-Chen/FADC) | C02低/高频引导的dilation 1/2/3混合 | 离散路由实验，没有连续自适应膨胀算子 |
| [Coordinate Attention，CVPR 2021](https://github.com/houqb/CoordAttention) | C03坐标门控 | 沿用前轮坐标注意力适配模块 |
| [SegFormer，NeurIPS 2021](https://research.nvidia.com/labs/lpr/publication/xie2021segformer/) | C06通道扩张、深度卷积、GELU和投影 | 借鉴Mix-FFN的局部混合；使用CNN分支，不引入完整Transformer |
| [FADE，ECCV 2022](https://github.com/poppinace/fade)、[论文](https://arxiv.org/abs/2207.10392) | C08结合浅层特征和解码logits预测输出采样偏移 | 借鉴编码/解码联合引导；采用grid_sample，不是FADE核重组或官方算子 |

C04/C05为解码宽度对照；C07为原型构建前的局部上下文残差；C09为局部尺度路由增加全局尺度先验。
后三者是本项目实验设计，不将其描述为已发表新方法。

## 权重追溯

实际路径、revision、大小、SHA256、来源记录在[pretrained_manifest.json](pretrained_manifest.json)。
下载采用固定revision并校验发布方LFS SHA256；构建模型再次校验本地SHA256，训练期间不联网。

- Small revision：`9a0c319ae4531d36df6ef11fc253fdc4b503628a`。
- Small SHA256：`d6888643d396f63c881f8658418609d6f8b98cc37653ee074b3b6769e4059ef5`。
- Base revision：`8e953102486d832f58e073cb4080f04001bb71ca`。
- Base SHA256：`afa69ae6094b5b04a48810de78c52c92d4cbb6d9436462f07bf36defb09e26c1`。
- Base文件350,300,064字节；存放于RS/pretrained/round12/convnext_base.dinov3_lvd1689m/。
- Small沿用RS/pretrained/round11/convnext_small.dinov3_lvd1689m/，没有重复下载。
- 许可证信息见对应下载目录README及manifest记录。
