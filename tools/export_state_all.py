"""Parallel export and audit of per-image module visualizations."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image,ImageDraw
from matplotlib import colormaps
from tools.export_tta import idle,lock
from tools.reporting import write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'state_outputs'
DATA=Path('/data2/tangyangpu/WorkSpace/RS/datasets')

def main():
    OUT.mkdir(exist_ok=True)
    singleton=lock(OUT/'.export.lock')
    if singleton is None:raise RuntimeError('An export is active')
    tasks=[('vaihingenR',0,1)]+[('potsdamR',i,4) for i in range(4)]
    active=[];failed=[]
    while tasks or active:
        for job in list(active):
            code=job['p'].poll()
            if code is not None:
                job['lease'].close();job['log'].close();active.remove(job)
                print('Finished',job['task'],'exit',code,flush=True)
                if code:failed.append(job['task'])
        for gpu,uuid in idle():
            if not tasks:break
            lease=lock(Path('/tmp')/f'glf_variants_{uuid}.lock')
            if lease is None:continue
            if (gpu,uuid) not in idle():lease.close();continue
            task=tasks.pop(0);dataset,index,count=task
            log=open(OUT/f'.{dataset}_{index}.log','w')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
            cmd=[sys.executable,str(ROOT/'tools/export_states.py'),'--dataset',dataset,'--data-root',str(DATA/dataset),'--shard-index',str(index),'--num-shards',str(count)]
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,pass_fds=(lease.fileno(),))
            active.append(dict(p=p,lease=lease,log=log,task=task))
            print('Started',task,'GPU',gpu,'PID',p.pid,flush=True)
        write_json(OUT/'status.json',dict(status='exporting',active=[j['task'] for j in active],pending=tasks,failed=failed))
        if tasks or active:time.sleep(5)
    if failed:
        write_json(OUT/'status.json',dict(status='failed',failed=failed));raise RuntimeError(failed)
    counts={};entries=[];png_count=0
    for dataset,tta_folder in [('vaihingenR','vaihigen'),('potsdamR','postdam')]:
        ids=sorted(p.stem for p in (DATA/dataset/'test/images_1024').glob('*.tif'))
        counts[dataset]=len(ids)
        for image_id in ids:
            folder=OUT/f'{dataset}__{image_id}'
            metadata=json.loads((folder/'metadata.json').read_text())
            files=list(folder.glob('*.png'));assert len(files)==38,(folder,len(files))
            assert len(metadata['maps'])==30
            assert (folder/'README.md').is_file() and (folder/'native_maps.npz').is_file()
            with np.load(folder/'native_maps.npz') as maps:
                assert len(maps.files)==30
                for key in maps.files:assert np.isfinite(maps[key]).all()
            with Image.open(folder/'42_final_tta_prediction.png') as im:a=np.array(im)
            with Image.open(ROOT/'tta_d4_results'/tta_folder/(image_id+'.png')) as im:b=np.array(im)
            assert np.array_equal(a,b)
            png_count+=len(files);entries.append((dataset,image_id,folder.name))
    legend=Image.new('RGB',(1000,180),'white');draw=ImageDraw.Draw(legend)
    names=['ImSurf','Building','LowVeg','Tree','Car','Clutter','Ignore']
    colors=[(255,255,255),(0,0,255),(0,255,255),(0,255,0),(255,204,0),(255,0,0),(0,0,0)]
    for i,(name,color) in enumerate(zip(names,colors)):
        x=i*140+5;draw.rectangle((x,5,x+35,35),fill=color,outline='black');draw.text((x,45),name,fill='black')
    gradient=(colormaps['viridis'](np.linspace(0,1,960))[None,:,:3]*255).round().astype('uint8')
    legend.paste(Image.fromarray(gradient).resize((960,25)),(20,90))
    draw.text((20,120),'0 / vmin',fill='black');draw.text((900,120),'1 / vmax',fill='black')
    draw.text((20,145),'RMS: limits in metadata; probabilities / routing weights / entropy: fixed [0,1]',fill='black')
    legend.save(OUT/'legend.png')
    lines=['# C10 关键模块逐图可视化','',
           '共617张输入：VaihingenR 113张、PotsdamR 504张；每张输入一个文件夹，每个文件夹38张PNG、README.md、metadata.json和native_maps.npz。','',
           '## 文件夹命名','',
           '`<数据集>__<原始图像ID>/`，两数据集同名图不会冲突。每个文件夹包含输入、GT、30张模块/概率热图、细化前预测、最终单次预测、最终TTA、叠加图、错误图和八图预览。','',
           '## 论文使用说明','',
           '- 模块特征来自eval模式、原始方向、scale=1的单次前向，最终单次预测与其对应。最终TTA图片复用先前24路测试结果，已逐像素核对一致。',
           '- 热图计算为通道RMS，即sqrt(mean(channel²))，并非每个通道原始特征、注意力权重或模块独立分类结果。只有route_weight图是实际softmax路由权重。',
           '- 每级适配前后热图共享该对图的1%–99%色标；残差及其他RMS热图各自使用1%–99%色标。不同模块/图像不能仅凭颜色亮度比较响应大小。概率、路由权重、归一化熵固定0–1。',
           '- 低分辨率特征着色后双线性放大至1024×1024；此操作不增加空间信息。输入图保持数据中原有三通道外观，未做颜色增强或裁剪。',
           '- native_maps.npz保存30张归一化前的原生二维float32数值；metadata.json包含原生尺寸、色标上下限、模块说明、权重与输入SHA256。可据此统一色标重绘。',
           '- 41_final_single_prediction.png为单次预测，42_final_tta_prediction.png为TTA最终预测；不要将TTA改进直接归因于某个模块。',
           '- 44_final_tta_error.png：红=错分，黑=正确，灰=ignore。辅助error_probability不是这张真实误差图。',
           '- 00_overview.png是低分辨率快速预览；论文排版使用各独立1024图，并参考每个文件夹README.md逐图标注。',
           '- 可视化反映当前训练模型的激活，不能替代消融实验来论证模块性能提升。','',
           '类别及色标图：[legend.png](legend.png)。模型源码：../geoseg/models/SFFNet/SFFNet.py；导出脚本：../tools/export_states.py。','',
           '## 图像索引','', '| 数据集 | 输入ID | 文件夹及逐图说明 |','|---|---|---|']
    for dataset,image_id,folder in entries:lines.append(f'| {dataset} | {image_id} | [{folder}]({folder}/README.md) |')
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    write_json(OUT/'status.json',dict(status='completed',folders=sum(counts.values()),datasets=counts,png_per_image=38,png_total=png_count,tta_pixels_verified=True,native_maps_verified=True))
    print('COMPLETED',counts,'PNGs',png_count,flush=True)
    singleton.close()

if __name__=='__main__':main()
