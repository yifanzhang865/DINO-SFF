"""Export faithful per-image C10 internal activations without modifying the model."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
from matplotlib import colormaps
import torch
from torch.nn import functional as F
from geoseg.models.SFFNet.SFFNet import SFFNet
from geoseg.datasets.vaihingen_dataset import val_aug
from tools.evaluation import sha256
from tools.reporting import write_json

ROOT=Path(__file__).resolve().parents[1]
PALETTE=[255,255,255,0,0,255,0,255,255,0,255,0,255,204,0,255,0,0,0,0,0]+[0]*(768-21)

def tensor_map(value):
    return value.detach().float()[0].cpu().numpy()

def energy(value):
    return tensor_map(value.float().square().mean(1).sqrt())

def save_label(values,path):
    image=Image.fromarray(values.astype('uint8'));image.putpalette(PALETTE);image.save(path)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['vaihingenR','potsdamR'],required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--num-shards',type=int,default=1)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if not 0<=args.shard_index<args.num_shards:parser.error('Invalid shard')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Expose one GPU')
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    entry=json.loads((ROOT/'manifest.json').read_text())[args.dataset]
    checkpoint=ROOT/entry['checkpoint'];assert sha256(checkpoint)==entry['sha256']
    model=SFFNet();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    model.cuda().eval()
    tta_folder='vaihigen' if args.dataset=='vaihingenR' else 'postdam'
    tta_report=json.loads((ROOT/'tta_d4_results'/tta_folder/'tta_metrics.json').read_text())
    assert tta_report['checkpoint_sha256']==entry['sha256'] and tta_report['saved_masks_confusion_verified']
    state_root=ROOT/'state_outputs';state_root.mkdir(exist_ok=True)
    status_path=state_root/f'.{args.dataset}_{args.shard_index}.json'
    ids=sorted(p.stem for p in (args.data_root/'test/images_1024').glob('*.tif'))[args.shard_index::args.num_shards]
    if args.limit:ids=ids[:args.limit]
    maps={};descriptions={};groups={};handles=[]
    def capture(name,description,mode='energy',group=None):
        def hook(module,inputs,output):
            value=output[0] if isinstance(output,tuple) else output
            maps[name]=energy(value) if mode=='energy' else tensor_map(value.sigmoid()[:,0])
            descriptions[name]=description;groups[name]=group
        return hook
    for i in range(4):
        stage=i+1
        before=f'10_stage{stage}_before_adapter';after=f'11_stage{stage}_after_adapter';delta=f'12_stage{stage}_adapter_residual'
        handles.append(model.backbone.encoder.stages[i].register_forward_hook(capture(before,f'DINOv3 Base stage{stage}输出，RestorationAdapter输入','energy',f'stage{stage}')))
        handles.append(model.backbone.adapters[i].register_forward_hook(capture(after,f'RestorationAdapter stage{stage}完整输出（含残差）；stage4尚未经过final_norm','energy',f'stage{stage}')))
        def residual_hook(module,inputs,output,name=delta,stage=stage):
            maps[name]=energy(output-inputs[0]);descriptions[name]=f'stage{stage}适配器实际添加的残差RMS(output-input)，不是注意力矩阵';groups[name]=None
        handles.append(model.backbone.adapters[i].register_forward_hook(residual_hook))
    modules=[('20_sampling_middle',model.sampling[1],'动态分组采样：stride16→stride8，与fine特征融合后的输出'),
             ('21_sampling_fine',model.sampling[0],'动态分组采样：stride8→stride4，与fine特征融合后的输出'),
             ('23_scale_fusion_branch',model.route_mix,'尺度路由加权混合后route_mix分支输出；实际以0.1倍加回fine特征'),
             ('24_prototype_fine',model.exchange[0],'深层原型注入浅层特征后的输出'),
             ('25_prototype_deep',model.exchange[1],'浅层信息回投深层原型后的输出'),
             ('26_deep_context',model.deep_context,'深层ObjectContext输出'),
             ('27_reverse_decoder',model.reverse,'ReverseDecoder返回的第一个张量z：粗到细反向细化特征'),
             ('28_fine_context',model.fine_context,'浅层ObjectContext输出'),
             ('29_detail_feature',model.detail,'margin/entropy引导的细节特征输出')]
    for name,module,desc in modules:handles.append(module.register_forward_hook(capture(name,desc)))
    def routing(module,inputs,output):
        weights=output.softmax(1)
        for i in range(4):
            name=f'22_route_weight_scale{i+1}'
            maps[name]=tensor_map(weights[:,i]);descriptions[name]=f'尺度路由softmax对stride{4*2**i}分支的权重（各像素四个权重之和为1）';groups[name]='probability'
    handles.append(model.scale_route.register_forward_hook(routing))
    total=len(ids)
    writer_pool=ThreadPoolExecutor(max_workers=6)
    def save_heat(heat,size,path):
        Image.fromarray(heat).resize(size,Image.Resampling.BILINEAR).save(path,compress_level=1)
    with torch.inference_mode():
        for index,image_id in enumerate(ids):
            maps.clear();descriptions.clear();groups.clear()
            name=f'{args.dataset}__{image_id}';destination=state_root/name
            if (destination/'metadata.json').exists():
                prior=json.loads((destination/'metadata.json').read_text())
                assert prior['checkpoint_sha256']==entry['sha256'] and (destination/'native_maps.npz').exists()
                assert all((destination/r['file']).exists() for r in prior['maps'])
                continue
            temporary=state_root/('.'+name+'.tmp')
            if temporary.exists():shutil.rmtree(temporary)
            temporary.mkdir()
            with Image.open(args.data_root/'test/images_1024'/(image_id+'.tif')) as im:original=im.convert('RGB')
            with Image.open(args.data_root/'test/masks_1024'/(image_id+'.png')) as im:label=np.array(im)
            image_array=np.array(original);normalized,_=val_aug(original,Image.fromarray(label))
            x=torch.from_numpy(normalized).permute(2,0,1)[None].float().cuda()
            output=model(x);height,width=label.shape
            prediction=output['logits'].argmax(1)[0].cpu().numpy()
            def logits_resize(logits):return F.interpolate(logits,size=(height,width),mode='bilinear',align_corners=False)
            coarse=logits_resize(output['errors'][0][1]).argmax(1)[0].cpu().numpy()
            probability=output['logits'].softmax(1)
            maps['33_final_confidence']=tensor_map(probability.max(1).values)
            descriptions['33_final_confidence']='单次前向最终softmax最大类别概率';groups['33_final_confidence']='probability'
            maps['34_final_entropy']=tensor_map(-(probability*probability.clamp_min(1e-6).log()).sum(1)/math.log(6))
            descriptions['34_final_entropy']='单次前向最终归一化预测熵，越大越不确定';groups['34_final_entropy']='probability'
            for key,tensor,desc in [('30_boundary_probability',output['boundary'],'辅助boundary头的sigmoid概率'),
                                    ('31_error_probability',output['errors'][0][0],'辅助error头的sigmoid概率，不是真实错误图'),
                                    ('32_car_probability',output['car'],'辅助car头的sigmoid概率，不是最终Car分类概率')]:
                maps[key]=tensor_map(tensor.sigmoid()[:,0]);descriptions[key]=desc;groups[key]='probability'
            original.save(temporary/'00_input.png');save_label(label,temporary/'01_ground_truth.png')
            save_label(coarse,temporary/'40_before_detail_prediction.png');save_label(prediction,temporary/'41_final_single_prediction.png')
            folder='vaihigen' if args.dataset=='vaihingenR' else 'postdam'
            source_tta=ROOT/'tta_d4_results'/folder/(image_id+'.png')
            with Image.open(source_tta) as im:
                tta=np.array(im);assert im.size==(width,height) and im.mode=='P'
                im.save(temporary/'42_final_tta_prediction.png')
                tta_rgb=im.convert('RGB')
            Image.blend(original,tta_rgb,0.45).save(temporary/'43_final_tta_overlay.png')
            errors=np.zeros((height,width,3),dtype=np.uint8)
            errors[(tta!=label)&(label<6)]=[255,0,0];errors[label==6]=[128,128,128]
            Image.fromarray(errors).save(temporary/'44_final_tta_error.png')
            rows=[];ranges={};writes=[]
            for group in [f'stage{i}' for i in range(1,5)]:
                vals=np.concatenate([value.ravel() for key,value in maps.items() if groups[key]==group])
                ranges[group]=np.percentile(vals,[1,99]).tolist()
            for key,values in maps.items():
                assert np.isfinite(values).all(),key
                group=groups[key]
                low,high=([0.,1.] if group=='probability' else ranges[group] if group else np.percentile(values,[1,99]).tolist())
                normalized=np.clip((values-low)/max(high-low,1e-12),0,1)
                heat=(colormaps['viridis'](normalized)[...,:3]*255).round().astype('uint8')
                writes.append(writer_pool.submit(save_heat,heat,(width,height),temporary/(key+'.png')))
                rows.append(dict(file=key+'.png',module=descriptions[key],native_shape=list(values.shape),vmin=low,vmax=high,
                                 normalization='fixed 0..1' if group=='probability' else 'shared stage pair p1/p99' if group else 'per-image p1/p99'))
            np.savez_compressed(temporary/'native_maps.npz',**maps)
            for future in writes:future.result()
            # Paper-selection contact sheet, never a replacement for the full-resolution panels.
            panels=[('00_input.png','Input'),('01_ground_truth.png','Ground truth'),('11_stage4_after_adapter.png','Stage4 + adapter'),
                    ('21_sampling_fine.png','Fine dynamic sampling'),('24_prototype_fine.png','Fine prototype exchange'),('29_detail_feature.png','Detail feature'),
                    ('41_final_single_prediction.png','Final single pass'),('42_final_tta_prediction.png','Final TTA')]
            sheet=Image.new('RGB',(4*256,2*280),'white');draw=ImageDraw.Draw(sheet)
            for j,(file,title) in enumerate(panels):
                with Image.open(temporary/file) as im:sheet.paste(im.convert('RGB').resize((256,256)),((j%4)*256,(j//4)*280+24))
                draw.text(((j%4)*256+5,(j//4)*280+5),title,fill='black')
            sheet.save(temporary/'00_overview.png')
            metadata=dict(dataset=args.dataset,image_id=image_id,checkpoint_sha256=entry['sha256'],best_epoch=entry['best_epoch'],
                          input_sha256=sha256(args.data_root/'test/images_1024'/(image_id+'.tif')),
                          input_size=[height,width],module_forward='single pass, original orientation, scale=1, eval, FP32',
                          final_tta_source=str(source_tta),final_tta_sha256=sha256(source_tta),maps=rows)
            write_json(temporary/'metadata.json',metadata)
            lines=[f'# {args.dataset} / {image_id}','',
                   '模块输出来自原始方向、scale=1的一次前向；最终TTA图复用已核验的D4×三尺度结果。',
                   '中间特征图不是语义分割图，也不是消融实验结果；不能仅依据颜色强弱判断模块效果。','',
                   '| 文件 | 内容 |','|---|---|',
                   '| 00_input.png | 模型归一化前的原始RGB输入 |',
                   '| 01_ground_truth.png | GT；ignore=6为黑色 |',
                   '| 00_overview.png | 八图预览拼图；论文排版请使用独立1024图 |']
            for row in rows:lines.append(f"| {row['file']} | {row['module']}；原生尺寸{row['native_shape']}；显示范围[{row['vmin']:.6g},{row['vmax']:.6g}] |")
            lines += ['| 40_before_detail_prediction.png | 细节修正之前的真实分类logits预测，双线性放大后argmax |',
                      '| 41_final_single_prediction.png | 单次前向最终预测，与本文件夹模块图对应 |',
                      '| 42_final_tta_prediction.png | 已完成的24路TTA最终预测 |',
                      '| 43_final_tta_overlay.png | 原图与TTA预测叠加，预测透明度0.45 |',
                      '| 44_final_tta_error.png | TTA与GT逐像素比较：红=错误、黑=正确、灰=ignore |','',
                      '特征热图为sqrt(mean(channel²))，viridis色图。每个stage的适配前/后共享该对图的1%–99%范围；其余RMS图各自使用1%–99%范围。概率、权重、熵固定0–1。颜色经过裁剪，仅用于显示。',
                      '所有PNG均为输入分辨率；低分辨率热图先着色再双线性放大，细节不代表新增空间信息。',
                      'native_maps.npz保存归一化前的原生二维float32数值，可统一色标重绘；metadata.json记录各图范围、尺寸、来源及权重SHA256。',
                      '类别颜色：ImSurf白、Building蓝、LowVeg青、Tree绿、Car黄、Clutter红。',
                      '本文件夹的模块分析针对训练所得模型，不代表模块独立分割性能或因果贡献。']
            (temporary/'README.md').write_text('\n\n'.join(lines[:6])+'\n'+'\n'.join(lines[6:])+'\n')
            if destination.exists():raise RuntimeError(f'Refusing to overwrite {destination}')
            temporary.rename(destination)
            write_json(status_path,dict(status='running',dataset=args.dataset,shard=args.shard_index,done=index+1,total=total,last_image=image_id))
            del output,x
    write_json(status_path,dict(status='completed',dataset=args.dataset,shard=args.shard_index,done=total,total=total))
    writer_pool.shutdown()
    for handle in handles:handle.remove()

if __name__=='__main__':main()
