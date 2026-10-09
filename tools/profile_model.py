"""Reproducible FP32 FLOPs and CUDA allocator memory measurements."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from collections import Counter

import torch
import ttach as tta
from torch.utils.flop_counter import FlopCounterMode
from geoseg.models.SFFNet.SFFNet import SFFNet
from tools.evaluation import Logits, sha256
from tools.reporting import write_json, atomic_text, now

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/model_profile'
CASES = ['infer_512', 'infer_1024', 'tta_1024', 'train_512_bs4']

class AuditedCounter(FlopCounterMode):
    def __init__(self):
        super().__init__(display=False)
        self.seen = Counter()
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.seen[str(func._overloadpacket)] += 1
        return super().__torch_dispatch__(func, types, args, kwargs)

def run_case(case):
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint = ROOT/'checkpoints/c10_dino_base_vaihingenR_best.pth'
    model = SFFNet()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.cuda()
    training = case.startswith('train')
    size = 512 if '512' in case else 1024
    batch = 4 if training else 1
    x = torch.randn(batch, 3, size, size, device='cuda')
    model.train(training)
    result = dict(case=case, measured_at=now(), gpu=torch.cuda.get_device_name(),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  cudnn_version=torch.backends.cudnn.version(), precision='FP32',
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  batch=batch, input_shape=list(x.shape),
                  parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                  buffer_bytes=sum(p.numel()*p.element_size() for p in model.buffers()),
                  checkpoint_bytes=checkpoint.stat().st_size, checkpoint_sha256=sha256(checkpoint))
    if training:
        from catalyst.contrib.optimizers import Lookahead
        from geoseg.losses.segmentation import compute_loss
        groups = [dict(params=[p for n,p in model.named_parameters() if n.startswith('backbone.')],lr=6e-5),
                  dict(params=[p for n,p in model.named_parameters() if not n.startswith('backbone.')],lr=6e-4)]
        optimizer = Lookahead(torch.optim.AdamW(groups, weight_decay=.01))
        # Match the training entrypoint, including its legacy Catalyst step wrapping.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
        labels = torch.randint(0, 7, (batch,size,size), device='cuda')
        def step():
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            loss = compute_loss(output, labels, 'c10_dino_base')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        result['scope']='forward + loss + backward + clipping + Lookahead AdamW step'
    else:
        forward = model
        if case.startswith('tta'):
            transforms=tta.Compose([tta.HorizontalFlip(),tta.VerticalFlip(),tta.Rotate90(angles=[90]),
                                   tta.Scale(scales=[.75,1.,1.5],interpolation='bilinear',align_corners=False)])
            assert len(transforms)==24
            forward=tta.SegmentationTTAWrapper(Logits(model),transforms,merge_mode='mean')
        @torch.inference_mode()
        def step():
            output = forward(x)
            # Return to keep the real output alive through the peak measurement.
            return output
        result['scope']='24-view multiscale D4 TTA' if case.startswith('tta') else 'full forward (including auxiliary heads)'
    # Initialize algorithms and (for training) all optimizer buffers before measuring.
    for _ in range(2):
        output=step()
        del output
    if training:
        optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    result['allocated_before_mib']=torch.cuda.memory_allocated()/2**20
    result['reserved_before_mib']=torch.cuda.memory_reserved()/2**20
    torch.cuda.reset_peak_memory_stats()
    times=[]
    for _ in range(5):
        start=time.perf_counter()
        output=step()
        torch.cuda.synchronize()
        times.append((time.perf_counter()-start)*1000)
        del output
    result.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                  median_ms=statistics.median(times),iteration_ms=times,warmup_iterations=2,measured_iterations=5)
    # Instrumentation is separate from memory and latency measurement.
    if not training:
        with torch.no_grad(), AuditedCounter() as counter:
            output=forward(x)
        counted={str(k):v for k,v in counter.get_flop_counts()['Global'].items()}
        result.update(flops=counter.get_total_flops(), gflops=counter.get_total_flops()/1e9,
                      gmacs=counter.get_total_flops()/2e9,flops_by_operator=counted,
                      uncounted_operator_calls={k:v for k,v in counter.seen.items() if k not in counted})
        assert result['flops']>0
        del output
    write_json(OUT/(case+'.json'),result)
    print(json.dumps(result,ensure_ascii=False),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',choices=CASES)
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if args.case:
        run_case(args.case)
        return
    OUT.mkdir(parents=True,exist_ok=True)
    # Independent processes avoid allocator contamination across scenarios.
    if not args.report_only:
        for case in CASES:
            subprocess.run([sys.executable,'-m','tools.profile_model','--case',case],cwd=ROOT,check=True)
    rows=[json.loads((OUT/(case+'.json')).read_text()) for case in CASES]
    first=rows[0]
    assert abs(rows[1]['flops']/rows[0]['flops']-4)<.02
    lines=['# C10 DINOv3 Base FLOPs 与显存实测','',f"测量时间：{now()}；GPU：{first['gpu']}；PyTorch：{first['torch_version']}。",'',
           f"参数量：**{first['parameters']:,}（{first['parameters']/1e6:.4f} M）**；FP32 参数内存：**{first['parameter_bytes']/2**20:.2f} MiB**；buffer：{first['buffer_bytes']/2**20:.4f} MiB；权重文件：{first['checkpoint_bytes']/2**20:.2f} MiB。",'',
           '| 场景 | Batch | 输入尺寸 | GMACs | GFLOPs | 峰值 allocated (MiB) | 峰值 reserved (MiB) | 中位耗时 (ms/次) |',
           '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        macs=f"{r['gmacs']:.3f}" if 'gmacs' in r else '未测'
        flops=f"{r['gflops']:.3f}" if 'gflops' in r else '未测'
        lines.append(f"| {r['case']} | {r['batch']} | {r['input_shape'][2]}×{r['input_shape'][3]} | {macs} | {flops} | {r['peak_allocated_mib']:.2f} | {r['peak_reserved_mib']:.2f} | {r['median_ms']:.2f} |")
    lines+=['','## 测量口径','',
        '- 完整模型、原版 Vaihingen 最佳权重，FP32，无 AMP。输入为固定 seed=42 的合成张量；训练标签包含 0–6。没有写回或修改最佳权重。',
        '- FLOPs 使用 PyTorch 自带 FlopCounterMode 实际执行统计，乘法与加法各算一次，即 1 MAC=2 FLOPs；1G=10^9。覆盖卷积和矩阵乘法（包括注意力的 bmm/mm/addmm）。',
        '- **这是卷积/矩阵乘法口径，并非全部算子的精确总运算数**：不计 bias 加法、归一化、激活、softmax、插值、grid_sample、池化、逐元素运算及 TTA 变换/融合。未计入算子的调用清单保存在 JSON。推理仍实际执行所有辅助头，未做部署裁剪。',
        '- TTA 为 D4×[0.75,1,1.5] 共 24 路，FLOPs 对实际完整 TTA 路径计数；显存也是实际 wrapper 测量。TTA 顺序执行，不是 24 张同时送入模型。',
        '- memory 指 GPU 显存。allocated 峰值为 PyTorch 张量分配峰值，包含模型、输入及中间结果；reserved 为缓存分配器预留峰值，包含可复用缓存。两者均不是仅增量激活内存，也不等于 nvidia-smi 的进程显存；CUDA context、外部库分配不保证包含。',
        '- 每个场景在独立进程测量：预热 2 次，清空空闲缓存、重置峰值后测 5 次；计时经 CUDA synchronize，不包含数据读取和 CPU→GPU 传输。FLOPs 统计在显存/耗时测量之后单独执行。',
        '- 训练使用原版多头损失、Lookahead AdamW、梯度裁剪 5，batch=4；峰值包含优化器状态及前向/反向/更新，5 次实测覆盖 Lookahead 的一个完整周期。训练 FLOPs 未测，不能直接按推理 FLOPs 相乘代替。',
        f"- cuDNN TF32={first['cudnn_allow_tf32']}，matmul TF32={first['matmul_allow_tf32']}；cudnn benchmark=False，deterministic=True。",'',
        '## 重现','',
        '```bash','CUDA_VISIBLE_DEVICES=4 MPLCONFIGDIR=/tmp/c10_profile_mpl /data2/tangyangpu/RS/bin/python -m tools.profile_model','```','',
        f"权重 SHA256：`{first['checkpoint_sha256']}`。各场景详细结果见同目录 JSON。",'']
    atomic_text(OUT/'PROFILE.md','\n'.join(lines))
    write_json(OUT/'summary.json',dict(status='completed',results=rows))

if __name__=='__main__':
    main()
