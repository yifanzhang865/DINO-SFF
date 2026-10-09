"""Refresh a comparison using only completed full-dataset evaluations."""
import json
import math
from pathlib import Path
from tools.reporting import atomic_text, now

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'experiments/vaihingen_ablation'
VARIANTS = {
    'baseline': '完整模型（已有最佳权重，重新测试）',
    'no_adapter': '移除四级 RestorationAdapter，恒等映射',
    'no_sampling': '移除两级 GroupedSampler 残差分支',
    'no_scale_route': '移除自适应尺度路由及 route_mix',
    'no_exchange': '移除双向 PrototypeExchange 及其辅助监督',
    'no_reverse': 'ReverseDecoder 替换为普通自顶向下融合，保留三级辅助监督和 Context',
    'no_detail': '移除细节纠错分支及 error/car 辅助监督，保留边界头',
}

def read(path):
    return json.loads(path.read_text()) if path.exists() else {}

def ratio(a,b):
    return a/b if b else None

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else None

def extended(result):
    if not result:
        return {}
    r = dict(result)
    cm = r['confusion_matrix']
    n = sum(map(sum,cm))
    actual = list(map(sum,cm))
    predicted = [sum(row[j] for row in cm) for j in range(6)]
    correct = sum(cm[i][i] for i in range(6))
    rows=[]
    for i, old in enumerate(r['per_class']):
        tp = cm[i][i]; fn = actual[i]-tp; fp = predicted[i]-tp; tn=n-tp-fn-fp
        row=dict(old, TP=tp, FP=fp, FN=fn, TN=tn, predicted_pixels=predicted[i],
                 Specificity=ratio(tn,tn+fp), NPV=ratio(tn,tn+fn),
                 FPR=ratio(fp,fp+tn), FNR=ratio(fn,fn+tp),
                 BinaryAccuracy=ratio(tp+tn,n),
                 MCC=ratio(tp*tn-fp*fn,math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))
        row['BalancedAccuracy']=mean([row['Recall'],row['Specificity']])
        rows.append(row)
    r['per_class']=rows
    r.update(PixelError=1-r['OA'], microPrecision=r['OA'], microRecall=r['OA'], microF1=r['OA'],
             microIoU=ratio(correct,2*n-correct), valid_pixels=n,
             MCC=ratio(n*correct-sum(a*p for a,p in zip(actual,predicted)),
                       math.sqrt((n*n-sum(p*p for p in predicted))*(n*n-sum(a*a for a in actual)))))
    for key in ['Precision','Recall','F1','Dice','IoU','Specificity','NPV','FPR','FNR','BalancedAccuracy']:
        r['macro6_'+key]=mean([x[key] for x in rows])
        r['weighted_'+key]=sum(x[key]*actual[i]/n for i,x in enumerate(rows) if x[key] is not None)
    return r

def fmt(x, percent=True):
    return '—' if x is None else f'{x*(100 if percent else 1):.4f}'

def table(headers, rows):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows]+['']

def render():
    status={v:read(OUT/v/'status.json') for v in VARIANTS}
    results={v:extended(read(OUT/v/'tta_metrics.json')) for v in VARIANTS}
    plain={v:extended(read(OUT/v/'no_tta_metrics.json')) for v in VARIANTS}
    baseline=results['baseline']
    lines=['# Vaihingen 单因素消融实验','',f'更新时间（UTC）：{now()}','',
        '未完成实验显示“—”，不以 smoke、部分测试或验证分数替代最终 TTA 指标。', '',
        '## 实验协议','',
        '- 数据：`/data2/tangyangpu/WorkSpace/RS/datasets/vaihingenR`；完整 train/test，ignore=6。',
        '- 消融模型从相同 SHA256 的 DINOv3 Base 预训练权重重新训练；不加载已训练的分割模型。',
        '- 每组 105 epochs，seed=42，batch=4，验证 batch=2，FP32，512 随机裁剪、mosaic=0.25；Lookahead AdamW，主干 LR=6e-5，其余=6e-4，weight decay=0.01，梯度裁剪=5，CosineAnnealingWarmRestarts T0=15/Tmult=2。',
        '- 每组只保存一个 `best_model.pth`，按无 TTA 验证五类 mIoU 严格改善覆盖；不保存 last 或每轮权重。完整模型复用已存最佳权重（原记录最佳 epoch=13），在当前数据上重新进行无 TTA/TTA 测试，未重新训练基线。',
        '- TTA：D4 × [0.75,1,1.5]，24 路逆变换对齐后 logits 平均。',
        '- **沿用原项目 test 集选择最佳权重，再在该集合测试，属于选模集表现，不是独立盲测。** 仅单次 seed，不报告多 seed 方差或显著性。',
        '- 主 mIoU/mF1/mPrecision/mRecall/mAcc/mDice 为前五类均值（排除 Clutter）；OA/FWIoU/Kappa/MCC/micro/weighted 为六类。F1=Dice，mAcc=mRecall；单标签 micro Precision/Recall/F1=OA。',
        '- 比例指标统一乘 100，Kappa/MCC 亦乘 100；Δ 是相对完整模型的百分点变化，负数表示下降。零分母以“—”表示；宏均值忽略未定义值。',
        '- 参数量按实际保留参数计数。端到端推理耗时含数据加载、指标累计和首次 CUDA 算子开销；多 GPU 同时运行时共享 I/O，不作为严格延迟基准。FLOPs、边界距离、校准误差、AP/AUC 尚未测量。', '',
        '## 消融定义与进度','']
    lines+=table(['实验','改动','状态','轮次','最佳轮','参数量'],[[v,desc,status[v].get('status','queued'),status[v].get('epoch','—'),status[v].get('best_epoch','—'),status[v].get('parameters','—')] for v,desc in VARIANTS.items()])
    keys=['OA','mIoU','mF1','mPrecision','mRecall','mAcc','mDice','FWIoU','Kappa','MCC','mIoU6','mF1_6','PixelError','microIoU']
    lines+=['## TTA 总体指标（%）','']
    lines+=table(['指标']+list(VARIANTS),[[k]+[fmt(results[v].get(k)) for v in VARIANTS] for k in keys])
    lines+=['## 六类补充指标（%）','']
    extra=[f'{prefix}_{k}' for prefix in ['macro6','weighted'] for k in ['Precision','Recall','F1','Dice','IoU','Specificity','NPV','FPR','FNR','BalancedAccuracy']]+['microPrecision','microRecall','microF1']
    lines+=table(['指标']+list(VARIANTS),[[k]+[fmt(results[v].get(k)) for v in VARIANTS] for k in extra])
    lines+=['## 相对基线与 TTA 增益（百分点）','']
    def diff(a,b,k):
        return fmt(a[k]-b[k]) if a.get(k) is not None and b.get(k) is not None else '—'
    lines+=table(['实验','无 TTA mIoU','TTA mIoU','TTA 增益','ΔmIoU','ΔmF1','ΔOA'],[[v,fmt(plain[v].get('mIoU')),fmt(results[v].get('mIoU')),diff(results[v],plain[v],'mIoU')]+[diff(results[v],baseline,k) for k in ['mIoU','mF1','OA']] for v in VARIANTS])
    lines+=['## 逐类指标（%）','']
    classkeys=['IoU','F1','Dice','Precision','Recall','Specificity','NPV','FPR','FNR','BalancedAccuracy','BinaryAccuracy','MCC']
    for v in VARIANTS:
        lines += [f'### {v}','']
        if not results[v]:
            lines += ['待完整 TTA 完成。','']; continue
        r=results[v]
        lines+=table(['类别']+classkeys,[[row['name']]+[fmt(row.get(k)) for k in classkeys] for row in r['per_class']])
        lines+=table(['类别','真实像素','预测像素','TP','FP','FN','TN'],[[row['name']]+[row[k] for k in ['support','predicted_pixels','TP','FP','FN','TN']] for row in r['per_class']])
        lines+=['混淆矩阵：行是真值，列是预测。','']
        names=[row['name'] for row in r['per_class']]
        lines+=table(['真值 / 预测']+names,[[names[i]]+row for i,row in enumerate(r['confusion_matrix'])])
        lines += [f"测试图数：{r['images']}；有效像素：{r['valid_pixels']}；权重 SHA256：`{r.get('checkpoint_sha256')}`；图像 ID SHA256：`{r.get('image_ids_sha256')}`。",'']
    lines+=['## 成本与产物','']
    lines+=table(['实验','无 TTA 秒','TTA 秒','TTA 图/秒','TTA ms/图','TTA 峰值显存 MiB','最佳权重'],[[v,fmt(plain[v].get('evaluation_seconds'),False)]+[fmt(results[v].get(k),False) for k in ['evaluation_seconds','images_per_second','milliseconds_per_image','peak_cuda_memory_mib']]+[status[v].get('best_checkpoint') or '待生成'] for v in VARIANTS])
    lines+=['各实验目录包含 `config.json`、`status.json`、`metrics.csv`、`train.log`、`no_tta_metrics.json`、`tta_metrics.json`；主调度日志为 `suite.log`。', '',
            '## 指标定义','',
            '每类按 one-vs-rest 计算：IoU=TP/(TP+FP+FN)，Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，Specificity=TN/(TN+FP)，NPV=TN/(TN+FN)，FPR=1−Specificity，FNR=1−Recall，BalancedAccuracy=(Recall+Specificity)/2。weighted 按真实类别像素频率加权。MCC 使用六类混淆矩阵的多分类公式；逐类 MCC 使用二分类公式。', '']
    atomic_text(OUT/'COMPARISON.md','\n'.join(lines))

if __name__=='__main__':
    render()
