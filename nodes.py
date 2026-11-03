"""嘉游SDRAW · 双输入工作流组件。"""
import os, sys, json, shutil, subprocess, tempfile, time, hashlib, threading
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
_RUN_LOCK = threading.Lock()
_CACHE_LOCK = threading.Lock()
_CACHE = OrderedDict()
_CACHE_LIMIT = 64 * 1024 * 1024


def interrupted():
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except ImportError:
        return
    throw_exception_if_processing_interrupted()


def run_process(args, directory, timeout=1800):
    with open(Path(directory)/'process.log', 'w+', encoding='utf8') as log:
        proc = subprocess.Popen(args, stdout=log, stderr=log, cwd=ROOT, shell=False)
        started = time.monotonic()
        try:
            while proc.poll() is None:
                interrupted()
                if time.monotonic()-started > timeout:
                    raise RuntimeError('嘉游SDRAW：处理超时')
                time.sleep(.1)
            if proc.returncode:
                log.seek(0); message=log.read()[-2500:]
                raise RuntimeError('嘉游SDRAW：'+message)
            log.seek(0)
            for line in log:
                try:
                    event=json.loads(line)
                    if 'timing' in event:print('[嘉游SDRAW 耗时] '+json.dumps(event,ensure_ascii=False),flush=True)
                except (ValueError,TypeError):pass
        finally:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=3)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()


def rgba(image):
    return np.concatenate((image, np.ones((*image.shape[:2],1), np.float32)),axis=2)


def thumb(image, max_edge):
    h,w=image.shape[:2]; scale=min(1,max_edge/max(h,w)); size=(max(1,round(w*scale)),max(1,round(h*scale)))
    return np.asarray(Image.fromarray(np.rint(np.clip(image,0,1)*255).astype(np.uint8)).resize(size,Image.Resampling.LANCZOS)).copy()


def estimate(source, reference, mode, directory):
    identity={'kind':'homography','matrix':[1,0,0,0,1,0,0,0,1]}
    if mode=='off': return identity
    if mode=='translation':
        p=Path(directory);h,w=source.shape[:2];rh,rw=reference.shape[:2]
        rgba(source).astype('<f4').tofile(p/'source.f32');np.rint(rgba(reference)*255).astype(np.uint8).tofile(p/'target.u8')
        (p/'job.json').write_text(json.dumps({'action':'align','width':w,'height':h,'referenceWidth':rw,'referenceHeight':rh}))
        run_process([os.environ.get('JAYOU_NODE_BIN') or shutil.which('node'),str(ROOT/'runtime/bridge.mjs'),str(p)],p)
        return json.loads((p/'alignment.json').read_text())
    sa,sb=thumb(source,1280),thumb(reference,1280)
    if mode=='lightglue':
        from .alignment import align
        pa,pb=Path(directory)/'align-a.png',Path(directory)/'align-b.png'
        Image.fromarray(sa).save(pa);Image.fromarray(sb).save(pb)
        # Run separately so Comfy interruption can stop model inference.
        result=Path(directory)/'alignment.json'
        run_process([sys.executable,str(ROOT/'alignment.py'),str(pa),str(pb),str(result)],directory)
        return json.loads(result.read_text())
    import cv2
    from .alignment import estimate_geometry
    detector=cv2.SIFT_create(nfeatures=4096)
    ka,da=detector.detectAndCompute(cv2.cvtColor(sa,cv2.COLOR_RGB2GRAY),None)
    kb,db=detector.detectAndCompute(cv2.cvtColor(sb,cv2.COLOR_RGB2GRAY),None)
    if da is None or db is None: raise RuntimeError('图像纹理不足，无法对齐')
    pairs=cv2.BFMatcher().knnMatch(da,db,k=2)
    good=[pair[0] for pair in pairs if len(pair)==2 and pair[0].distance<.75*pair[1].distance]
    a=np.float32([ka[m.queryIdx].pt for m in good]);b=np.float32([kb[m.trainIdx].pt for m in good])
    result=estimate_geometry(a,b,(sa.shape[1],sa.shape[0]),(sb.shape[1],sb.shape[0]));result['method']='SIFT + MAGSAC++'
    return result


def decode_response(data, width, height, node):
    with tempfile.TemporaryDirectory(prefix='jayou-mask-') as tmp:
        p=Path(tmp);(p/'response.json').write_text(json.dumps(data));(p/'job.json').write_text(json.dumps({'action':'decode','width':width,'height':height}))
        run_process([node,str(ROOT/'runtime/bridge.mjs'),tmp],tmp,120)
        result=np.zeros((height,width),np.uint8)
        for name in json.loads((p/'decoded.json').read_text()):
            with Image.open(p/name) as im:
                im=im.convert('RGBA').resize((width,height),Image.Resampling.BILINEAR)
                a=np.asarray(im,dtype=np.float32); mask=np.rint(a[:,:,0]*a[:,:,3]/255).astype(np.uint8)
            np.maximum(result,mask,out=result)
        return result


def segment(image, prompt, node):
    import requests,io
    token=os.environ.get('JAYOU_SAM_TOKEN','');endpoint=os.environ.get('JAYOU_SAM_ENDPOINT','https://ai.gitee.com/v1/images/segmentation');model=os.environ.get('JAYOU_SAM_MODEL','sam3')
    if not token: raise RuntimeError('所选自动范围需要配置 JAYOU_SAM_TOKEN；也可连接 MASK 并选择替代模式')
    f=io.BytesIO();Image.fromarray(image).save(f,format='PNG');payload=f.getvalue()
    key=hashlib.sha256(payload+(endpoint+model+prompt+token).encode()).hexdigest()
    with _CACHE_LOCK:
        if key in _CACHE: _CACHE.move_to_end(key);return _CACHE[key].copy()
    interrupted()
    try:
        response=requests.post(endpoint,headers={'Authorization':'Bearer '+token},files={'image':('image.png',payload,'image/png')},data={'model':model,'prompt':prompt},timeout=(15,180),allow_redirects=False)
        if response.status_code!=200: raise RuntimeError(f'SAM 服务返回 HTTP {response.status_code}')
        data=response.json()
    except requests.RequestException as e: raise RuntimeError('SAM 网络请求失败') from None
    result=decode_response(data,image.shape[1],image.shape[0],node)
    with _CACHE_LOCK:
        while _CACHE and sum(m.nbytes for m in _CACHE.values())+result.nbytes>_CACHE_LIMIT:_CACHE.popitem(last=False)
        if result.nbytes<=_CACHE_LIMIT:_CACHE[key]=result.copy()
    return result


CONCEPTS={'window':['window','outdoor view through window','outdoor view through open doorway'],'sky':['sky'],'bed':['bed','duvet','pillows','bed runner','bed sheet','blanket','objects on bed'],'floor':['floor'],'wall':['wall']}
EXCLUSIONS=['bedside table','objects on bedside table']


def selections(source,reference,alignment,scopes,directory,node):
    import cv2
    a,b=thumb(source,2560),thumb(reference,2560);h,w=a.shape[:2];rh,rw=b.shape[:2];p=Path(directory)
    np.concatenate((a,np.full((h,w,1),255,np.uint8)),axis=2).tofile(p/'guide.u8');np.concatenate((b,np.full((rh,rw,1),255,np.uint8)),axis=2).tofile(p/'reference-guide.u8')
    yy,xx=np.mgrid[:h,:w];u=(xx+.5)/w;v=(yy+.5)/h;m=np.array(alignment['matrix']).reshape(3,3);d=m[2,0]*u+m[2,1]*v+m[2,2]
    mx=((m[0,0]*u+m[0,1]*v+m[0,2])/d*rw-.5).astype(np.float32);my=((m[1,0]*u+m[1,1]*v+m[1,2])/d*rh-.5).astype(np.float32)
    parts=[]
    for scope in scopes:
        both=[];ex=[]
        for im in (a,b):
            mask=np.zeros(im.shape[:2],np.uint8)
            for word in CONCEPTS[scope]:np.maximum(mask,segment(im,word,node),out=mask)
            excluded=np.zeros_like(mask)
            if scope=='bed' and mask.any():
                for word in EXCLUSIONS:np.maximum(excluded,segment(im,word,node),out=excluded)
            if mask.mean()/255>.98:raise RuntimeError('自动选区范围异常，请检查输入或改用 MASK')
            both.append(mask);ex.append(excluded)
        if not any(x.any() for x in both):print(f'[嘉游SDRAW] 未找到 {scope}，已跳过');continue
        excluded=np.maximum(ex[0],cv2.remap(ex[1],mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0))
        names={'source':scope+'-a.u8','reference':scope+'-b.u8','excluded':scope+'-exclude.u8'}
        for key,value in [('source',both[0]),('reference',both[1]),('excluded',excluded)]:value.tofile(p/names[key])
        parts.append(names)
    return {'width':w,'height':h,'referenceWidth':rw,'referenceHeight':rh,'components':parts}


class JayouSDRAW:
    CATEGORY='嘉游'
    FUNCTION='process'
    RETURN_TYPES=('IMAGE','MASK')
    RETURN_NAMES=('image','mask')
    DESCRIPTION='嘉游SDRAW 双输入工作流组件。'

    @classmethod
    def INPUT_TYPES(cls):
        r={'image_a':('IMAGE',),'image_b':('IMAGE',),
           '方式':(['参数反求','保边颜色变换'],),
           '对齐':(['SIFT + MAGSAC++','LightGlue + MAGSAC++','平移','关闭'],),
           '蒙版用途':(['替代自动选区','与自动选区取交集','与自动选区取并集'],)}
        for name,default in [('高光细节补偿',False),('暗部细节补偿',False),('大区域参考补偿',False),('窗户窗外门外',True),('天空',False),('床及床品',False),('地面',False),('墙面',False),('自动差异',False)]:r[name]=('BOOLEAN',{'default':default})
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
        return {'required':r,'optional':{'mask':('MASK',)}}

    @classmethod
    def IS_CHANGED(cls,**kwargs):
        return hashlib.sha256('|'.join(os.environ.get(k,'') for k in ['JAYOU_SAM_TOKEN','JAYOU_SAM_ENDPOINT','JAYOU_SAM_MODEL','JAYOU_NODE_BIN']).encode()).hexdigest()

    def process(self,image_a,image_b,mask=None,**kwargs):
        spec=self.INPUT_TYPES()['required'];opts={}
        for key,(kind,*rest) in spec.items():
            if key in ('image_a','image_b'):continue
            config=rest[0] if rest else {};value=kwargs.get(key,config.get('default',kind[0] if isinstance(kind,list) else None))
            if isinstance(kind,list):
                if value not in kind:raise ValueError(key+' 无效')
            elif kind=='BOOLEAN':
                if not isinstance(value,bool):raise ValueError(key+' 必须为布尔值')
            else:
                if not isinstance(value,(int,float)) or not np.isfinite(value) or value<config['min'] or value>config['max'] or kind=='INT' and int(value)!=value:raise ValueError(key+' 超出范围')
            opts[key]=value
        for image in (image_a,image_b):
            if not isinstance(image,torch.Tensor) or image.ndim!=4 or image.shape[-1]!=3 or min(image.shape[:3])<1:raise ValueError('IMAGE 必须是 [B,H,W,3]')
            if not torch.isfinite(image).all() or image.min()<0 or image.max()>1:raise ValueError('IMAGE 需要有限的 0–1 RGB 值')
            if image.shape[1]*image.shape[2]>100_000_000:raise ValueError('单图超过 1 亿像素')
        batch=image_a.shape[0]
        if image_b.shape[0] not in (1,batch):raise ValueError('参考批次数需为 1 或与原图一致')
        if mask is not None:
            if mask.ndim==2:mask=mask.unsqueeze(0)
            if mask.ndim==4 and mask.shape[1]==1:mask=mask[:,0]
            if mask.ndim!=3 or mask.shape[0] not in (1,batch) or min(mask.shape)<1 or not torch.isfinite(mask).all() or mask.min()<0 or mask.max()>1:raise ValueError('MASK 必须是有效的 [H,W] 或 [B,H,W]，范围 0–1')
        node=os.environ.get('JAYOU_NODE_BIN') or shutil.which('node')
        if not node:raise RuntimeError('请安装 Node.js 22+ 或设置 JAYOU_NODE_BIN')
        o={'method':{'参数反求':'parameters','保边颜色变换':'color'}[opts['方式']], 'alignment':{'关闭':'off','平移':'translation','SIFT + MAGSAC++':'sift','LightGlue + MAGSAC++':'lightglue'}[opts['对齐']], 'maskMode':{'替代自动选区':'replace','与自动选区取交集':'intersect','与自动选区取并集':'union'}[opts['蒙版用途']], 'edge':{'保持分割结果':'none','自适应RGB贴合':'adaptive','硬边':'hard','柔边':'soft'}[opts['边缘类型']]}
        mapping={'highlights':'高光细节补偿','shadows':'暗部细节补偿','content':'大区域参考补偿','difference':'自动差异','expand':'扩展收缩像素','feather':'羽化像素','fillHoles':'小孔洞修复','holeArea':'最大孔洞面积','smooth':'颜色柔和过渡','regional':'分区精细匹配','colorRefinement':'精细颜色映射','enhance':'色彩与对比增强','strength':'整体强度','recoveryStrength':'补偿强度','contentStrength':'内容补偿强度','detailProtection':'原有纹理保留','analysisSize':'分析长边'}
        o.update({k:opts[v] for k,v in mapping.items()});o['analysisSize']=int(o['analysisSize'])
        o['localLight']={'enabled':opts['局部光影迁移'],'highlights':opts['受光增强'],'shadows':opts['阴影塑形'],'scale':{'整体照明':'broad','均衡':'balanced','局部层次':'local'}[opts['光影尺度']],'color':1 if opts['受光色彩迁移'] else 0}
        scope_names={'window':'窗户窗外门外','sky':'天空','bed':'床及床品','floor':'地面','wall':'墙面'};scopes=[k for k,v in scope_names.items() if opts[v]]
        outputs=[];masks=[]
        while not _RUN_LOCK.acquire(timeout=.1):interrupted()
        try:
            for idx in range(batch):
                interrupted()
                with tempfile.TemporaryDirectory(prefix='jayou-sdraw-') as tmp:
                    p=Path(tmp);a=image_a[idx].detach().cpu().float().numpy();b=image_b[0 if image_b.shape[0]==1 else idx].detach().cpu().float().numpy();h,w=a.shape[:2];rh,rw=b.shape[:2]
                    if o['alignment']=='off' and abs(np.log((w/h)/(rw/rh)))>.01:raise ValueError('关闭对齐时，两图宽高比必须一致')
                    alignment=estimate(a,b,o['alignment'],tmp)
                    rgba(a).astype('<f4').tofile(p/'source.f32');np.rint(np.clip(rgba(b),0,1)*255).astype(np.uint8).tofile(p/'target.u8')
                    job={'width':w,'height':h,'referenceWidth':rw,'referenceHeight':rh,'options':o,'alignment':alignment,'alignmentResolved':True}
                    if o['content'] and scopes and not(mask is not None and o['maskMode']=='replace'):job['selection']=selections(a,b,alignment,scopes,tmp,node)
                    if mask is not None:
                        m=mask[0 if mask.shape[0]==1 else idx].detach().cpu().float().numpy();im=Image.fromarray(np.rint(m*255).astype(np.uint8));scale=min(1,2560/max(w,h));mw,mh=max(1,round(w*scale)),max(1,round(h*scale));np.asarray(im.resize((mw,mh),Image.Resampling.BILINEAR)).tofile(p/'input-mask.u8');job['mask']={'width':mw,'height':mh}
                    (p/'job.json').write_text(json.dumps(job));run_process([node,str(ROOT/'runtime/bridge.mjs'),tmp],tmp)
                    out=np.fromfile(p/'output.f32',dtype='<f4').reshape(h,w,3);coverage=np.fromfile(p/'output-mask.f32',dtype='<f4').reshape(h,w)
                    outputs.append(torch.from_numpy(out));masks.append(torch.from_numpy(coverage))
        finally:_RUN_LOCK.release()
        return torch.stack(outputs),torch.stack(masks)
