"""Atomic machine-readable and Markdown reports, with checkpoint-bound TTA."""
import csv
import io
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
VARIANTS = {"c10_dino_base": "C10 DINOv3 Base"}

SUMMARY_PREFIX = "glf_potsdam_c10_best"
METRICS = ["OA","mIoU","mF1","mPrecision","mRecall","mAcc","mDice","FWIoU","Kappa","mIoU6","mF1_6"]
PROTOCOL = ("数值为百分比（Kappa 也乘100展示）。mIoU/mF1/mPrecision/mRecall/mAcc/mDice取前五类，"
            "排除Clutter；OA/FWIoU/Kappa使用全部六类；ignore_label=6。mAcc=宏平均Recall，Dice=F1。"
            "TTA=D4×[0.75,1,1.5]，24次logits平均，不保存预测图。"
            "沿用test作为验证集选择最佳权重，因此这些结果不是独立盲测。")

def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def atomic_text(path,text):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix="."+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def write_json(path,value):
    atomic_text(path,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")

def read_json(path,default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default

def fmt(value):
    return "—" if value is None else f"{100*value:.4f}"

def model_lines(row):
    t=row.get("tta") or {}
    lines=[f"## {row['model']} — {VARIANTS[row['model']]}","",
           f"最佳验证轮次：{row.get('best_epoch')}；验证mIoU：{fmt(row.get('best_val_miou'))}；"
           f"状态：{row['status']}","",
           f"最佳权重：{row.get('best_checkpoint') or '待生成'}",""]
    if not t:
        return lines+["TTA 尚未完成。",""]
    lines += ["| "+ " | ".join(METRICS)+" |",
              "|"+ "|".join(["---:"]*len(METRICS))+"|",
              "| "+" | ".join(fmt(t.get(k)) for k in METRICS)+" |","",
              "| 类别 | IoU | F1 | Precision | Recall | Dice | 像素数 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in t["per_class"]:
        lines.append("| "+r["name"]+" | "+" | ".join(fmt(r[k]) for k in ("IoU","F1","Precision","Recall","Dice"))+f" | {r['support']} |")
    lines += ["",f"Checkpoint SHA256: {t['checkpoint_sha256']}",
              f"测试图像：{t['images']}；split：{t['split']}；seed：{t['seed']}",""]
    return lines

def render_model(row,path):
    protocol = PROTOCOL
    if (row.get('tta') or {}).get('transforms') == 8:
        protocol = protocol.replace('D4×[0.75,1,1.5]，24次logits平均', '单尺度D4（scale=1），8次logits平均')
    if (row.get('tta') or {}).get('saved_prediction_images'):
        protocol = protocol.replace('不保存预测图。', '保存带类别调色板的PNG预测图，像素值为0–5。')
    atomic_text(path,protocol+"\n\n"+"\n".join(model_lines(row))+"\n")
