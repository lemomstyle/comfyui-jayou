import {deflateSync,inflateSync,gunzipSync} from 'node:zlib';
import {imageMime} from './media.mjs';
export const giteeSegmentEndpoint='https://ai.gitee.com/v1/images/segmentation';
export function isGiteeSegment(s){return s.provider==='gitee'||s.endpoint?.replace(/\/$/,'')===giteeSegmentEndpoint;}
// COCO RLE uses column-major pixels and alternating background/foreground runs.
export function decodeRle(mask){
 const [h,w]=mask.size||[];if(!Number.isSafeInteger(h)||!Number.isSafeInteger(w)||h<1||w<1||h*w>67108864)throw Error('Gitee 蒙版尺寸无效或超过 6400 万像素解码容量');
 let runs=mask.counts;
 if(typeof runs==='string'){
  if(runs.length>33554432)throw Error('Gitee RLE 过长');
  const counts=[];let pos=0;
  while(pos<runs.length){let value=0,shift=0,c;
   do{if(pos>=runs.length||shift>30)throw Error('Gitee RLE 编码损坏');c=runs.charCodeAt(pos++)-48;if(c<0||c>63)throw Error('Gitee RLE 字符无效');value+=(c&31)*2**shift;shift+=5;}while(c&32);
   if(c&16)value-=2**shift;
   if(counts.length>2)value+=counts[counts.length-2];counts.push(value);
  }runs=counts;
 }
 if(!Array.isArray(runs)||runs.length>h*w+1||runs.some(x=>!Number.isSafeInteger(x)||x<0)||runs.reduce((a,b)=>a+b,0)!==h*w)throw Error('Gitee RLE 长度与蒙版尺寸不匹配');
 const pixels=new Uint8Array(w*h);let offset=0;
 for(let i=0;i<runs.length;i++){const end=offset+runs[i];if(i%2)for(let j=offset;j<end;j++)pixels[(j%h)*w+Math.floor(j/h)]=255;offset=end;}
 return {width:w,height:h,pixels};
}
function crc(bytes){let n=0xffffffff;for(const b of bytes){n^=b;for(let i=0;i<8;i++)n=(n>>>1)^((n&1)?0xedb88320:0);}return (n^0xffffffff)>>>0;}
function chunk(name,data){const type=Buffer.from(name),head=Buffer.alloc(4),tail=Buffer.alloc(4);head.writeUInt32BE(data.length);tail.writeUInt32BE(crc(Buffer.concat([type,data])));return Buffer.concat([head,type,data,tail]);}
export function maskPng({width,height,pixels}){const head=Buffer.alloc(13);head.writeUInt32BE(width);head.writeUInt32BE(height,4);head[8]=8;const scan=Buffer.alloc((width+1)*height);for(let y=0;y<height;y++)scan.set(pixels.subarray(y*width,(y+1)*width),y*(width+1)+1);return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',head),chunk('IDAT',deflateSync(scan)),chunk('IEND',Buffer.alloc(0))]);}
// Read upload geometry rather than assume the service follows COCO's [height,width].
export function segmentationImageSize(bytes){
 if(bytes.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10]))&&bytes.length>=24)return {width:bytes.readUInt32BE(16),height:bytes.readUInt32BE(20)};
 if(bytes[0]!==255||bytes[1]!==216)return null;
 let i=2;while(i+4<bytes.length){if(bytes[i++]!==255)continue;while(bytes[i]===255)i++;const marker=bytes[i++];if(marker===217||marker===218)break;if(marker===1||(marker>=208&&marker<=215))continue;const length=bytes.readUInt16BE(i);if(length<2||i+length>bytes.length)break;if([192,193,194,195,197,198,199,201,202,203,205,206,207].includes(marker)&&length>=7)return {height:bytes.readUInt16BE(i+3),width:bytes.readUInt16BE(i+5)};i+=length;}return null;
}
export function normalizeGiteeRle(rle,expected){
 const [a,b]=rle.size||[];
 if(expected&&a===expected.width&&b===expected.height&&a!==b)return {...rle,size:[b,a]};
 return rle;
}
export function decodeGiteeMask(mask,expected=null){
 if(!mask||typeof mask!=='object')throw Error('Gitee 返回的区域缺少像素蒙版');
 if(String(mask.encoding).toLowerCase()==='coco_rle_base64'){
  const encoded=mask.counts??mask.data;if(typeof encoded!=='string'||encoded.length>24000000||!/^[A-Za-z0-9+/]+={0,2}$/.test(encoded))throw Error('Gitee COCO_RLE_base64 数据无效');
  let bytes=Buffer.from(encoded,'base64');
  if(bytes[0]===31&&bytes[1]===139)bytes=gunzipSync(bytes,{maxOutputLength:32000000});
  else if(bytes[0]===120&&((bytes[0]*256+bytes[1])%31===0))bytes=inflateSync(bytes,{maxOutputLength:32000000});
  const text=new TextDecoder('utf-8',{fatal:true}).decode(bytes);let rle={size:mask.size,counts:text};
  if(text.trimStart().startsWith('{')){const value=JSON.parse(text);rle={size:value.size??mask.size,counts:value.counts};if(mask.size&&JSON.stringify(mask.size)!==JSON.stringify(rle.size))throw Error('Gitee 蒙版内外尺寸不一致');}
  return 'data:image/png;base64,'+maskPng(decodeRle(normalizeGiteeRle(rle,expected))).toString('base64');
 }
 if(String(mask.encoding).toLowerCase()==='rle')return 'data:image/png;base64,'+maskPng(decodeRle(normalizeGiteeRle(mask,expected))).toString('base64');
 if(String(mask.encoding).toLowerCase()==='base64'){
  const value=mask.data??mask.counts;if(typeof value!=='string'||value.length>12000000)throw Error('Gitee base64 蒙版无效');
  const raw=value.startsWith('data:')?value.slice(value.indexOf(',')+1):value;if(!/^[A-Za-z0-9+/]+={0,2}$/.test(raw))throw Error('Gitee base64 蒙版编码无效');const bytes=Buffer.from(raw,'base64');return `data:${imageMime(bytes)};base64,${bytes.toString('base64')}`;
 }
 throw Error('Gitee 蒙版编码暂不支持：'+String(mask.encoding).slice(0,30)+'；需要真实响应示例进行适配');
}
export async function giteeSegment(body,s,validateImage,fetcher){
 if(!s.key)throw Error('请配置 Gitee SAM3 Access Token');
 const source=validateImage(body.source),comma=source.indexOf(','),mime=source.slice(5,source.indexOf(';'));
 const expected=segmentationImageSize(Buffer.from(source.slice(comma+1),'base64'));
 const form=new FormData();form.set('model',s.model||'sam3');form.set('prompt',body.prompt);form.set('image',new Blob([Buffer.from(source.slice(comma+1),'base64')],{type:mime}),'source.'+(mime==='image/jpeg'?'jpg':mime.split('/')[1]));
 let response,data;
 try{response=await fetcher(s.endpoint,{method:'POST',headers:{Authorization:'Bearer '+s.key,Accept:'application/json'},body:form,redirect:'error',signal:AbortSignal.timeout(180000)});if(!response.ok)throw Error(`Gitee SAM3 返回 ${response.status}：${response.status===401?'令牌无效':response.status===429?'请求限额或额度不足':'请检查模型权限与服务状态'}`);data=await response.json();}catch(e){if(e.name==='TimeoutError'||e.name==='AbortError')throw Error('Gitee SAM3 分割超时，请稍后重试');if(e instanceof TypeError)throw Error('Gitee SAM3 网络连接失败，请检查网络后重试');throw e;}
 if(!Array.isArray(data.segments))throw Error('Gitee SAM3 未返回 segments 蒙版数组');
 if(!data.segments.length){if(body.allowEmpty===true)return {masks:[],model:s.model||'sam3',scores:[],empty:true};throw Error('Gitee SAM3 未找到该对象，请调整英文对象描述');}
 if(data.segments.length>64)throw Error('Gitee 返回区域过多，请缩小对象描述范围');
 // All returned instances correspond to the submitted concept. Never substitute detection boxes for masks.
 return {masks:data.segments.map(x=>decodeGiteeMask(x.mask,expected)),model:s.model||'sam3',scores:data.segments.map(x=>Number.isFinite(x.confidence)?x.confidence:null)};
}
