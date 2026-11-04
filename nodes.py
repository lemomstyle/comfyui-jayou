import json
import threading
import numpy as np
import torch
from .common import interrupted, chunks
from .pipeline import process as execute
from .registration import estimate, warp

_LOCK = threading.Lock()


def validate_image(image):
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] != 3 or min(image.shape) < 1:
        raise ValueError('IMAGE 必须是 [B,H,W,3]')
    if image.shape[1]*image.shape[2] > 100_000_000 or max(image.shape[1:3]) >= 32767:
        raise ValueError('单图最多 1 亿像素，单边必须小于 32767')
    for member in image:
        for top, bottom in chunks(member.shape[0], member.shape[1]):
            strip = member[top:bottom]
            if not torch.isfinite(strip).all() or strip.min() < 0 or strip.max() > 1:
                raise ValueError('IMAGE 必须为有限的 0–1 RGB 数据')


def array(image):
    return image.detach().to(device='cpu', dtype=torch.float32).contiguous().numpy()


class JayouSDRAWPython:
    CATEGORY = '嘉游'
    FUNCTION = 'process'
    RETURN_TYPES = ('IMAGE', 'MASK')
    RETURN_NAMES = ('image', 'mask')
    DESCRIPTION = '嘉游SDRAW Python 双输入组件。区域输入由外部 MASK 提供。'

    @classmethod
    def INPUT_TYPES(cls):
        r={'image_a':('IMAGE',),'image_b':('IMAGE',),
           '方式':(['参数反求','保边颜色变换'],),
           '对齐':(['SIFT + MAGSAC++','LightGlue + MAGSAC++','平移','关闭'],),
           '蒙版用途':(['替代自动选区','与自动选区取交集','与自动选区取并集'],)}
        for name,default in [('高光细节补偿',False),('暗部细节补偿',False),('大区域参考补偿',False),('自动差异',False)]:r[name]=('BOOLEAN',{'default':default})
        r['边缘类型']=(['保持分割结果','自适应RGB贴合','硬边','柔边'],)
        r['扩展收缩像素']=('FLOAT',{'default':0.,'min':-12.,'max':12.,'step':.5})
        r['羽化像素']=('FLOAT',{'default':0.,'min':0.,'max':12.,'step':.5})
        r['小孔洞修复']=('BOOLEAN',{'default':False})
        r['最大孔洞面积']=('INT',{'default':64,'min':1,'max':4096})
        for name in ['颜色柔和过渡','分区精细匹配','精细颜色映射','色彩与对比增强']:r[name]=('BOOLEAN',{'default':False})
        for name in ['整体强度','补偿强度','内容补偿强度','原有纹理保留']:r[name]=('FLOAT',{'default':1.,'min':0.,'max':1.,'step':.01})
        r['分析长边']=(['512','768','1024'],{'default':'512'})
        r['局部光影迁移']=('BOOLEAN',{'default':False})
        r['受光增强']=('FLOAT',{'default':.8,'min':0.,'max':1.,'step':.05})
        r['阴影塑形']=('FLOAT',{'default':.8,'min':0.,'max':1.,'step':.05})
        r['光影尺度']=(['整体照明','均衡','局部层次'],{'default':'均衡'})
        r['受光色彩迁移']=('BOOLEAN',{'default':False})
        r['质感精修']=('BOOLEAN',{'default':False})
        r['质感强度']=('FLOAT',{'default':.5,'min':0.,'max':1.,'step':.05})
        return {'required':r,'optional':{'mask':('MASK',)}}

    def process(self, image_a, image_b, mask=None, **kwargs):
        spec = self.INPUT_TYPES()['required']; opts = {}
        unknown = set(kwargs)-set(spec)
        if unknown: raise ValueError('未知参数：'+', '.join(sorted(unknown)))
        for key, (kind, *rest) in spec.items():
            if key in ('image_a', 'image_b'): continue
            config = rest[0] if rest else {}
            value = kwargs.get(key, config.get('default', kind[0] if isinstance(kind, list) else None))
            if isinstance(kind, list):
                if value not in kind: raise ValueError(key+' 无效')
            elif kind == 'BOOLEAN':
                if not isinstance(value, bool): raise ValueError(key+' 必须为布尔值')
            elif (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
                  or value < config['min'] or value > config['max'] or kind == 'INT' and int(value) != value):
                raise ValueError(key+' 超出范围')
            opts[key] = value
        validate_image(image_a); validate_image(image_b)
        batch, h, w, _ = image_a.shape
        if image_b.shape[0] not in (1, batch): raise ValueError('参考图批次必须为 1 或与原图相同')
        if mask is not None:
            if not isinstance(mask, torch.Tensor): raise ValueError('MASK 必须为 Tensor')
            if mask.ndim == 2: mask = mask.unsqueeze(0)
            if mask.ndim == 4 and mask.shape[1] == 1: mask = mask[:, 0]
            if mask.ndim != 3 or min(mask.shape) < 1 or mask.shape[0] not in (1, batch):
                raise ValueError('MASK 必须是 [H,W] 或 [B,H,W]，批次为 1 或与原图相同')
            for member in mask:
                for top, bottom in chunks(member.shape[0], member.shape[1]):
                    part = member[top:bottom]
                    if not torch.isfinite(part).all() or part.min() < 0 or part.max() > 1:
                        raise ValueError('MASK 必须是有限的 0–1 数据')
        if opts['大区域参考补偿'] and mask is None and not opts['自动差异']:
            raise ValueError('大区域补偿需要外接 MASK（例如 SAM3 输出），或开启自动差异')
        if (opts['大区域参考补偿'] and mask is not None and opts['蒙版用途'] == '与自动选区取交集'
                and not opts['自动差异']):
            raise ValueError('取交集需要开启自动差异；只使用 SAM3 MASK 时请选择替代自动选区')
        while not _LOCK.acquire(timeout=.1): interrupted()
        try:
            out = torch.empty((batch, h, w, 3), dtype=torch.float32, device='cpu')
            coverage = torch.empty((batch, h, w), dtype=torch.float32, device='cpu')
            for i in range(batch):
                interrupted()
                a = array(image_a[i]); b = array(image_b[0 if image_b.shape[0] == 1 else i])
                m = array(mask[0 if mask.shape[0] == 1 else i]) if mask is not None else None
                result, used, diagnostics = execute(a, b, m, opts)
                out[i].copy_(torch.from_numpy(result)); coverage[i].copy_(torch.from_numpy(used))
                print('[嘉游SDRAW Python] '+json.dumps(diagnostics, ensure_ascii=False), flush=True)
                del result, used
            return out, coverage
        finally:
            _LOCK.release()


class JayouSDRAWAlignPython:
    CATEGORY = '嘉游'
    FUNCTION = 'process'
    RETURN_TYPES = ('IMAGE', 'IMAGE', 'MASK')
    RETURN_NAMES = ('原图', '对齐参考图', '有效重叠范围')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image_a': ('IMAGE',), 'image_b': ('IMAGE',),
                '对齐方式': (['SIFT + MAGSAC++', 'LightGlue + MAGSAC++', '平移'],)}}

    def process(self, image_a, image_b, 对齐方式='SIFT + MAGSAC++'):
        if 对齐方式 not in self.INPUT_TYPES()['required']['对齐方式'][0]: raise ValueError('未知对齐方式')
        validate_image(image_a); validate_image(image_b)
        batch, h, w, _ = image_a.shape
        if image_b.shape[0] not in (1, batch): raise ValueError('参考图批次必须为 1 或与原图相同')
        out = torch.empty((batch, h, w, 3), dtype=torch.float32)
        masks = torch.empty((batch, h, w), dtype=torch.float32)
        for i in range(batch):
            a = array(image_a[i]); b = array(image_b[0 if image_b.shape[0] == 1 else i])
            matrix = estimate(a, b, 对齐方式)
            result, valid = warp(b, a, matrix)
            out[i].copy_(torch.from_numpy(result)); masks[i].copy_(torch.from_numpy(valid))
        return image_a, out, masks
