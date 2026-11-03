import {fitLocalLight} from './core/local-light-fit.js';
import fs from 'node:fs';
import {performance} from 'node:perf_hooks';
const started=performance.now();let lastStage=started;
const timing=label=>{const now=performance.now();console.log(JSON.stringify({timing:label,seconds:(now-lastStage)/1000,totalSeconds:(now-started)/1000}));lastStage=now;};
import path from 'node:path';
import {render} from './core/engine.js';
import {renderSupport} from './core/sharpen.js';
import {resizeSample,solveParameters,solveColorTransfer,enhanceParameters,refineRegionalParameters,correspondenceWeights,alignPair} from './core/same-image-transfer.js';
import {refineMatchedColors} from './core/color-refinement-fit.js';
import {warpReference} from './core/image-alignment.js';
import {analyzeMatchRegions,recoveryToneField} from './core/match-regions.js';
import {recoveryMask,effectiveCoverage} from './core/reference-recovery.js';
import {processPairedEdges} from './core/paired-edge-processing.js';
import {mergeDifference,unionMask} from './core/compensation-scopes.js';
import {compensationToneMask} from './core/compensation-edges.js';
import {processMaskEdges} from './core/adaptive-mask.js';
import {sampleMask} from './core/mask-resample.js';
import {decodeGiteeMask} from './service/gitee-segment.mjs';
const dir=process.argv[2],j=JSON.parse(fs.readFileSync(path.join(dir,'job.json'),'utf8'));
const read=(name,Type=Uint8Array)=>{const b=fs.readFileSync(path.join(dir,name));return new Type(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};
const save=(name,value)=>fs.writeFileSync(path.join(dir,name),Buffer.from(value.buffer,value.byteOffset,value.byteLength));
const progress=(phase,progress)=>console.log(JSON.stringify({phase,progress}));
try{
 if(j.action==='decode'){
  const data=JSON.parse(fs.readFileSync(path.join(dir,'response.json'),'utf8'));if(!Array.isArray(data.segments)||data.segments.length>64)throw Error('Invalid segmentation response');
  const files=data.segments.map((s,i)=>{const url=decodeGiteeMask(s.mask,{width:j.width,height:j.height}),name=`mask-${i}.png`;fs.writeFileSync(path.join(dir,name),Buffer.from(url.split(',')[1],'base64'));return name;});fs.writeFileSync(path.join(dir,'decoded.json'),JSON.stringify(files));
 }else if(j.action==='align'){
 const source=read('source.f32',Float32Array),target=read('target.u8',Uint8ClampedArray),sa=resizeSample(source,j.width,j.height,192),tb=warpReference(target,j.referenceWidth,j.referenceHeight,sa.width,sa.height,{kind:'homography',matrix:[1,0,0,0,1,0,0,0,1]}).pixels,r=alignPair(Uint8ClampedArray.from(sa.pixels,v=>v*255),tb,sa.width,sa.height);
 if(r.score<.5)throw Error('平移对齐未找到可靠对应点');fs.writeFileSync(path.join(dir,'alignment.json'),JSON.stringify({kind:'homography',matrix:[1,0,r.x,0,1,r.y,0,0,1]}));
 }else{
 const o=j.options,w=j.width,h=j.height,tw=j.referenceWidth,th=j.referenceHeight,source=read('source.f32',Float32Array),target=read('target.u8',Uint8ClampedArray);
 if(source.length!==w*h*4||target.length!==tw*th*4)throw Error('Input dimensions do not match data');
 let alignment=j.alignment;
 if(o.alignment==='translation'&&!j.alignmentResolved){const sa=resizeSample(source,w,h,192),tb=warpReference(target,tw,th,sa.width,sa.height,{kind:'homography',matrix:[1,0,0,0,1,0,0,0,1]}).pixels,ab=Uint8ClampedArray.from(sa.pixels,v=>v*255),r=alignPair(ab,tb,sa.width,sa.height);if(r.score<.5)throw Error('平移对齐未找到可靠对应点');alignment={kind:'homography',matrix:[1,0,r.x,0,1,r.y,0,0,1]};}
 const guideAt=(mw,mh)=>{const out=new Uint8ClampedArray(mw*mh*4);for(let y=0;y<mh;y++)for(let x=0;x<mw;x++){const k=(Math.min(h-1,Math.floor((y+.5)*h/mh))*w+Math.min(w-1,Math.floor((x+.5)*w/mw)))*4;for(let c=0;c<4;c++)out[(y*mw+x)*4+c]=source[k+c]*255;}return out;};
 const a=resizeSample(source,w,h,o.analysisSize),b=resizeSample(target,tw,th,o.analysisSize),{pixels,valid}=warpReference(b.pixels,b.width,b.height,a.width,a.height,alignment),regions=analyzeMatchRegions(a.pixels,pixels,a.width,a.height,valid);
 if(regions.fitValid.filter(Boolean).length<Math.max(16,valid.length*.05))throw Error('可靠对应区域不足，请调整对齐方式');
 let selection=null;
 if(j.selection){const s=j.selection;selection={...s,mask:new Uint8Array(s.width*s.height),guide:read('guide.u8'),referenceGuide:read('reference-guide.u8'),components:s.components.map(p=>({sourceMask:read(p.source),referenceMask:read(p.reference),excluded:read(p.excluded)}))};}
 if(o.content&&o.difference)selection=mergeDifference(selection,regions.contentMask,a.width,a.height);
 if(selection){selection.guide ||= guideAt(selection.width,selection.height);selection=processPairedEdges(selection,alignment,w,h,{mode:o.edge,expand:o.expand,feather:o.feather,fillHoles:o.fillHoles,holeArea:o.holeArea});}
 // External MASK is already in source coordinates. It controls content only.
 if(j.mask&&o.content){const mw=selection?.width||j.mask.width,mh=selection?.height||j.mask.height,rawManual=sampleMask(read('input-mask.u8'),j.mask.width,j.mask.height,{width:mw,height:mh}),manual=processMaskEdges(rawManual,guideAt(mw,mh),mw,mh,w,h,{mode:o.edge,expand:o.expand,feather:o.feather,fillHoles:o.fillHoles,holeArea:o.holeArea}),automatic=selection?.mask||new Uint8Array(mw*mh);let merged=manual;
  if(o.maskMode==='intersect')merged=Uint8Array.from(automatic,(v,i)=>Math.round(v*manual[i]/255));else if(o.maskMode==='union'){merged=automatic.slice();unionMask(merged,manual);}
  selection={mask:merged,width:mw,height:mh};
 }
 timing('准备与选区');
 const opts={valid:regions.fitValid,progress:p=>progress(p.phase,p.progress)};
 let result=o.method==='parameters'?solveParameters(a.pixels,pixels,a.width,a.height,opts):solveColorTransfer(a.pixels,pixels,a.width,a.height,opts);
 timing('基础反求');
 if(o.enhance)result=enhanceParameters(a.pixels,pixels,a.width,a.height,result,opts);
 timing('色彩对比增强');
 if(o.regional)result=refineRegionalParameters(a.pixels,pixels,a.width,a.height,result,opts);
 timing('分区匹配');
 if(o.colorRefinement)result=refineMatchedColors(a.pixels,pixels,a.width,a.height,result,opts);
 timing('精细颜色映射');
 if(o.localLight?.enabled)result=fitLocalLight(a.pixels,pixels,a.width,a.height,result,{...opts,...o.localLight});
 timing('局部光影');
 const params=result.params;
 if(o.highlights||o.shadows||o.content){const rendered=render(a.pixels,a.width,a.height,params,{layers:result.layers,outputFormat:'float32'}),confidence=correspondenceWeights(a.pixels,pixels,a.width,a.height,valid),mask=recoveryMask(rendered,pixels,a.width,a.height,valid,confidence,{adaptive:true,source:a.pixels});
 params.referenceRecovery={adaptive:2,detailProtection:o.detailProtection,toneField:recoveryToneField(rendered,pixels,a.width,a.height),mask,maskWidth:a.width,maskHeight:a.height,pixels:target,width:tw,height:th,matrix:alignment.matrix,strength:o.recoveryStrength,highlights:o.highlights?1:0,shadows:o.shadows?1:0,contentStrength:o.content&&selection?o.contentStrength:0,...(selection?{contentMask:selection.mask,contentWidth:selection.width,contentHeight:selection.height,contentToneMask:o.smooth?compensationToneMask(selection.mask,selection.width,selection.height):undefined}:{})};}
 timing('补偿准备');
 const imageFd=fs.openSync(path.join(dir,'output.f32'),'w'),maskFd=fs.openSync(path.join(dir,'output-mask.f32'),'w'),baseParams={...params};delete baseParams.referenceRecovery;
 const halo=Math.max(3,renderSupport(params,w,h,1,result.layers)),rows=Math.max(1,Math.floor(1048576/w));
 try{for(let start=0;start<h;start+=rows){const end=Math.min(h,start+rows),top=Math.max(0,start-halo),bottom=Math.min(h,end+halo),chunk=source.subarray(top*w*4,bottom*w*4),options={layers:result.layers,outputFormat:'float32',spatial:{x:0,y:top,width:w,height:h}},capture=params.referenceRecovery&&params.localLight?.strength?new Float32Array(chunk.length):null,out=render(chunk,w,bottom-top,params,{...options,preRecoveryOutput:capture}),base=capture||(params.referenceRecovery&&(params.referenceRecovery.highlights||params.referenceRecovery.shadows)?render(chunk,w,bottom-top,baseParams,options):null),coverage=new Float32Array((end-start)*w),image=new Float32Array((end-start)*w*3);
  for(let y=start;y<end;y++)for(let x=0;x<w;x++){const k=(y-top)*w+x,i=k*4,q=(y-start)*w+x;for(let c=0;c<3;c++)image[q*3+c]=Math.max(0,Math.min(1,chunk[i+c]+(out[i+c]-chunk[i+c])*o.strength));if(params.referenceRecovery)coverage[q]=effectiveCoverage(params.referenceRecovery,(x+.5)/w,(y+.5)/h,base?base[i]:0,base?base[i+1]:0,base?base[i+2]:0,{data:chunk,w,h:bottom-top,inputScale:1},k)*o.strength;}
  fs.writeSync(imageFd,Buffer.from(image.buffer));fs.writeSync(maskFd,Buffer.from(coverage.buffer));progress('输出',end/h);
 }}finally{fs.closeSync(imageFd);fs.closeSync(maskFd);}
 timing('全尺寸渲染与写出');
 fs.writeFileSync(path.join(dir,'result.json'),JSON.stringify({initial:result.initial,error:result.error,alignment,localLightAccepted:result.localLightAccepted,localLightError:result.localLightError,excluded:regions.excludedFraction}));
 }
}catch(e){console.error(e.message);process.exitCode=1;}
