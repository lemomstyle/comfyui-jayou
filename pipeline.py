"""Fit at bounded size, apply to original-size strips; no reference replacement by default."""
import time
import cv2
import numpy as np
from .common import chunks, resize, resize_to, interrupted
from .color import render, smooth
from .registration import estimate, warp
from .fitting import fit_parameters, correspondence, loss
from .fields import fit_affine, apply_affine, fit_gain, apply_light
from .refinement import fit_lut, apply_lut
from .texture_finish import finish
from .masks import process_edges, fill_small_holes, recovery


def process(a, b, mask, o, matrix=None):
    start = time.monotonic()
    h, w = a.shape[:2]
    # Full precision reference is retained; only fitting uses a small preview.
    matrix = estimate(a, b, o['对齐']) if matrix is None else matrix
    target, overlap = warp(b, a, matrix)
    x = resize(a, int(o['分析长边'])); y = resize_to(target, x.shape[:2]); v = resize_to(overlap, x.shape[:2])
    weights, mismatch = correspondence(x, y, v)
    if np.count_nonzero(weights > .1) < max(16, weights.size*.05):
        raise ValueError('可靠对应区域不足，请检查对齐和两张图片')
    params = None; mapping = None; regional = None; lut = None; light = None
    diagnostics = {}
    if o['方式'] == '参数反求':
        params = fit_parameters(x, y, v)
        base = render(x, params)
        if o['色彩与对比增强']:
            candidate = fit_parameters(x, y, v, enhance=True)
            enhanced = render(x, candidate)
            accepted = loss(enhanced, y, weights, True, True) < loss(base, y, weights, True, True)*.995
            diagnostics['enhancement_accepted'] = accepted
            if accepted: params, base = candidate, enhanced
    else:
        mapping = fit_affine(x, y, weights)
        base = apply_affine(mapping, x)
        if o['色彩与对比增强']:
            candidate = fit_lut(base, y, weights, 7)
            enhanced = apply_lut(candidate, base)
            accepted = loss(enhanced, y, weights, True, True) < loss(base, y, weights, True, True)*.995
            diagnostics['enhancement_accepted'] = accepted
            if accepted: lut, base = candidate, enhanced
    if o['分区精细匹配']:
        candidate = fit_affine(base, y, weights, 7)
        enhanced = apply_affine(candidate, base)
        if loss(enhanced, y, weights, True) < loss(base, y, weights, True)*.995:
            regional, base = candidate, enhanced
        diagnostics['regional_accepted'] = regional is not None
    # Keep enhancement LUT separate from final color refinement to preserve order.
    enhancement_lut = lut; lut = None
    if o['精细颜色映射']:
        candidate = fit_lut(base, y, weights)
        enhanced = apply_lut(candidate, base)
        if loss(enhanced, y, weights, True) < loss(base, y, weights, True)*.995:
            lut, base = candidate, enhanced
        diagnostics['color_refinement_accepted'] = lut is not None
    if o['局部光影迁移']:
        side = {'整体照明': 12, '均衡': 24, '局部层次': 32}[o['光影尺度']]
        candidate = (fit_gain(base, y, weights, 8), fit_gain(base, y, weights, side))
        enhanced = apply_light(candidate, base, o['受光增强'], o['阴影塑形'], int(o['受光色彩迁移']))
        clipping = np.mean(((enhanced > 1)&(y < .98)&(base <= 1))|((enhanced < 0)&(y > .02)&(base >= 0)))
        if loss(enhanced, y, weights, True) < loss(base, y, weights, True)*.995 and clipping < .01:
            light = candidate
        diagnostics['local_light_accepted'] = light is not None
    selection = np.zeros((h, w), np.float32)
    if o['大区域参考补偿']:
        automatic = resize_to(smooth(.25, .65, mismatch), (h, w)) if o['自动差异'] else selection
        if mask is not None:
            manual = resize_to(mask, (h, w))
            mode = o['蒙版用途']
            selection = manual if mode == '替代自动选区' else (np.minimum(manual, automatic) if mode == '与自动选区取交集' else np.maximum(manual, automatic))
        else: selection = automatic
        if o['小孔洞修复']: selection = fill_small_holes(selection, o['最大孔洞面积'])
    diagnostics['fit_seconds'] = round(time.monotonic()-start, 3)
    output = np.empty_like(a); coverage = np.empty((h, w), np.float32)
    # Filter support includes morphological, guided-filter and Gaussian neighborhoods.
    halo = int(np.ceil(abs(o['扩展收缩像素'])+3*o['羽化像素']))+24
    for top, bottom in chunks(h, w):
        lo, hi = max(0, top-halo), min(h, bottom+halo)
        original = a[lo:hi]
        q = render(original, params) if params is not None else apply_affine(mapping, original, lo, (h, w))
        if enhancement_lut is not None: q = apply_lut(enhancement_lut, q)
        if regional is not None: q = apply_affine(regional, q, lo, (h, w))
        if lut is not None: q = apply_lut(lut, q)
        if light is not None: q = apply_light(light, q, o['受光增强'], o['阴影塑形'], int(o['受光色彩迁移']), lo, (h, w))
        selected = process_edges(selection[lo:hi], original, o['边缘类型'], o['扩展收缩像素'], o['羽化像素']) if o['大区域参考补偿'] else selection[lo:hi]
        q, cov = recovery(q, original, target[lo:hi], overlap[lo:hi], selected, o)
        amount = o['整体强度']
        q = original+(q-original)*amount
        output[top:bottom] = np.clip(q[top-lo:bottom-lo], 0, 1)
        coverage[top:bottom] = cov[top-lo:bottom-lo]*amount
    if o.get('质感精修', False) and o.get('质感强度', .5) > 0 and o['整体强度'] > 0:
        diagnostics['texture_finish'] = finish(output, target, a, overlap, o.get('质感强度', .5)*o['整体强度'])
    diagnostics['total_seconds'] = round(time.monotonic()-start, 3)
    diagnostics['mask_fraction'] = float(coverage.mean())
    interrupted()
    return output, coverage, diagnostics
