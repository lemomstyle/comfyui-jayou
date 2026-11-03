import {guidedMaskEdges,processMaskEdges,validateEdgeOptions} from './adaptive-mask.js';
import {pairedSelection,subtractSelection} from './paired-selection.js';
import {unionMask} from './compensation-scopes.js';
// Each photograph guides only its own mask. No source-only refinement after union.
export function processPairedEdges(selection,alignment,originalWidth,originalHeight,options){
 options=validateEdgeOptions(options);
 const {width:w,height:h,guide}=selection,out=new Uint8Array(w*h);
 for(const part of selection.components||[{mask:selection.mask}]){
  let value;
  if(part.sourceMask&&part.referenceMask){
   const a=guidedMaskEdges(part.sourceMask,guide,w,h,options.mode),b=guidedMaskEdges(part.referenceMask,selection.referenceGuide,selection.referenceWidth,selection.referenceHeight,options.mode);
   const united=pairedSelection(a,b,w,h,selection.referenceWidth,selection.referenceHeight,alignment);
   value=processMaskEdges(united,guide,w,h,originalWidth,originalHeight,{...options,mode:'none'});
  }else value=processMaskEdges(part.mask,guide,w,h,originalWidth,originalHeight,options);
  if(part.excluded)value=subtractSelection(value,part.excluded);
  unionMask(out,value);
 }
 return {...selection,mask:out};
}
