import {composeBrushWeight} from './brush-composite.js';
export function finalMaskWeight(coverage,inverted=false,brushes=null,index=0){const value=Math.max(0,Math.min(1,coverage)),base=inverted?1-value:value;return composeBrushWeight(base,brushes,index);}
// Pixel masks are immutable/shared; edits replace them. Small editable structures
// are cloned so changing one mask cannot alter another mask's recipe.
export function duplicateLayer(layer,id){
 const {mask,instructionId,...recipe}=layer;
 const copy=structuredClone(recipe);
 return {...copy,...(mask?{mask}:{}),id,name:(layer.name||'蒙版')+' 副本'};
}
export function validateMaskInversion(layer){if(layer.inverted!==undefined&&typeof layer.inverted!=='boolean')throw Error('蒙版反选状态非法');}

// Complementing every max/min operation preserves the exact complement of the
// current composite, including soft brush edges. Replacement keeps history safe.
export function invertLayerSelection(layer){return {...layer,inverted:!layer.inverted,postBrushOps:layer.postBrushOps?.map(o=>({...o,erase:!o.erase}))};}
