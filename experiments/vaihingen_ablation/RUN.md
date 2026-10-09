# Vaihingen 消融实验运行说明

主报告：[COMPARISON.md](COMPARISON.md)。调度器每 30 秒更新；缺失结果用“—”表示。只有各组完成全部 105 轮训练，并对最佳权重完成全量无 TTA、24 路 TTA 测试后才有最终结果。

六个变体：`no_adapter`、`no_sampling`、`no_scale_route`、`no_exchange`、`no_reverse`、`no_detail`。定义和指标口径见主报告。每个变体只保存一个 `best_model.pth`；完整基线使用仓库已有权重，不复制。Smoke 检查没有保存权重，不计入正式指标。

## 当前后台任务

- Python：`/data2/tangyangpu/RS/bin/python`
- 调度器 PID：见 `suite.pid`
- GPU：3、4、5、6、7，每 GPU 同时只有本任务的一个子进程。
- 日志：`suite.log`、各变体目录的 `train.log`
- 进度：各变体的 `status.json`、`metrics.csv`
- 最终状态：所有任务退出后生成 `suite_status.json`，任一失败则标记 failed。
- 调度器使用独立进程会话运行，脱离本次终端仍继续；但不保证机器重启或管理员终止后继续。
- 为遵守只保存最佳权重，不保存 optimizer/scheduler 的恢复检查点。意外中断不能精确续训；调度器检测到未完成实验的已有最佳权重会拒绝覆盖，需人工处理。

## 查看进度

在项目根目录执行：

```bash
tail -n 20 experiments/vaihingen_ablation/suite.log
tail -n 5 experiments/vaihingen_ablation/no_adapter/metrics.csv
cat experiments/vaihingen_ablation/no_adapter/status.json
```

手动刷新报告（无需 GPU）：

```bash
PYTHONPATH=. /data2/tangyangpu/RS/bin/python -m experiments.vaihingen_ablation.report
```

## 单组重现实验

示例（请在 GPU 空闲且目标输出目录没有需要保留的结果时运行；不要与正在运行的同名实验同时运行）：

```bash
CUDA_VISIBLE_DEVICES=3 /data2/tangyangpu/RS/bin/python train_supervision.py -c config/vaihingen/no_adapter.py
```

只对消融最佳权重重复 TTA 测试：

```bash
CUDA_VISIBLE_DEVICES=3 /data2/tangyangpu/RS/bin/python vaihingen_test.py \
  -c config/vaihingen/no_adapter.py -t d4 --no-save \
  -o experiments/vaihingen_ablation/no_adapter/retest
```

改变 config 文件名即可运行其他变体。完整基线的配置和权重保持原路径。

## 验证记录

`verification.json` 记录完整基线严格加载、六组前向/反向、所有保留参数有限梯度检查。`smoke_no_reverse/status.json` 和 `smoke_tta_metrics.json` 记录两批真实训练/验证及单图 24 路 TTA，仅验证流程。`suite_manifest.json` 记录预训练来源、源代码 SHA256、train/test 图像 ID；每组 `launch.json` 保存实际命令、GPU。

参数量：完整模型 89,437,565；no_adapter 89,114,357；no_sampling 89,398,765；no_scale_route 89,390,713；no_exchange 89,399,151；no_reverse 89,422,013；no_detail 89,341,461。

注意：该模型主体是 DINOv3 ConvNeXt Base，因此这些解码器/适配器消融的总参数量变化较小。no_reverse 使用普通自顶向下解码器作为必要替代；no_exchange 和 no_detail 的模块专属辅助损失随模块一并移除，其他损失系数不变，辅助分割损失仍对剩余辅助输出取平均。
