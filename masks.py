"""External mask processing: preserve interiors; refine only a boundary band."""
import cv2
import numpy as np
from .color import LUMA, smooth
from .common import interrupted


def fill_small_holes(mask, area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask < .5).astype(np.uint8), 8)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    fill = (stats[:, cv2.CC_STAT_AREA] <= area)
    fill[border] = False; fill[0] = False
    out = mask.copy(); out[fill[labels]] = 1
    interrupted()
    return out


def guided(guide, p, radius=4, eps=.002):
    # Color covariance distinguishes equal-luminance boundaries as well.
    k = (2*radius+1,)*2
    mean = lambda x: cv2.boxFilter(x, -1, k, borderType=cv2.BORDER_REFLECT_101)
    mi, mp = mean(guide), mean(p)
    covariance = np.empty((*p.shape, 3, 3), np.float32)
    cross = mean(guide*p[..., None])-mi*mp[..., None]
    for i in range(3):
        for j in range(i, 3):
            value = mean(guide[..., i]*guide[..., j])-mi[..., i]*mi[..., j]
            covariance[..., i, j] = value
            covariance[..., j, i] = value
        covariance[..., i, i] += eps
    coefficients = np.linalg.solve(covariance, cross[..., None])[..., 0]
    intercept = mp-np.sum(coefficients*mi, axis=-1)
    return np.sum(mean(coefficients)*guide, axis=-1)+mean(intercept)


def process_edges(mask, source, mode, expand, feather):
    m = mask.astype(np.float32, copy=True)
    if expand:
        # Subpixel radius interpolates the two neighboring integer radii.
        radius = abs(expand); lo = int(np.floor(radius)); hi = int(np.ceil(radius))
        fn = cv2.dilate if expand > 0 else cv2.erode
        def mor(r):
            if not r: return m.copy()
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
            return fn(m, kernel, borderType=cv2.BORDER_REPLICATE)
        m = mor(lo)*(1-radius+lo)+mor(hi)*(radius-lo)
    if mode == '硬边': m = (m >= .5).astype(np.float32)
    if mode == '自适应RGB贴合':
        # Locked interior prevents texture in bedding from punching new holes.
        kernel = np.ones((7, 7), np.uint8)
        inside = cv2.erode(m, kernel, borderType=cv2.BORDER_REPLICATE)
        outside = cv2.dilate(m, kernel, borderType=cv2.BORDER_REPLICATE)
        refined = np.clip(guided(source, m), 0, 1)
        m = np.where(inside >= .999, m, np.where(outside <= .001, m, refined))
    sigma = max(feather, .6 if mode == '柔边' else 0)
    if sigma:
        radius = max(1, int(np.ceil(3*sigma)))
        m = cv2.GaussianBlur(m, (2*radius+1,)*2, sigma, borderType=cv2.BORDER_REFLECT_101)
    return np.clip(m, 0, 1).astype(np.float32)


def recovery(base, source, reference, valid, selection, options):
    sy, by, ty = source @ LUMA, base @ LUMA, reference @ LUMA
    highlight = np.zeros(sy.shape, np.float32)
    shadow = highlight.copy()
    if options['高光细节补偿']:
        # Require loss evidence in the original, not just a newly bright grade.
        highlight = smooth(.90, .995, sy)*smooth(.02, .16, sy-ty)
    if options['暗部细节补偿']:
        shadow = (1-smooth(.005, .065, sy))*smooth(.02, .15, ty-sy)
    loss_mask = np.maximum(highlight, shadow)
    if options['原有纹理保留'] and loss_mask.any():
        sx = cv2.Sobel(sy, cv2.CV_32F, 1, 0)/8; syy = cv2.Sobel(sy, cv2.CV_32F, 0, 1)/8
        tx = cv2.Sobel(ty, cv2.CV_32F, 1, 0)/8; tyy = cv2.Sobel(ty, cv2.CV_32F, 0, 1)/8
        sgrad, tgrad = np.hypot(sx, syy), np.hypot(tx, tyy)
        cosine = (sx*tx+syy*tyy)/(sgrad*tgrad+1e-8)
        reliable = smooth(.005, .04, sgrad)*smooth(.3, .9, cosine)
        # Texture protection limits automatic detail replacement, never grading
        # or an explicitly selected external content region.
        loss_mask *= 1-reliable*options['原有纹理保留']
    loss_mask *= options['补偿强度']
    content = selection*options['内容补偿强度'] if options['大区域参考补偿'] else np.zeros_like(selection)
    alpha = np.maximum(content, loss_mask)*valid
    target = reference
    if options['颜色柔和过渡'] and np.any(alpha):
        delta = base-reference
        low = cv2.GaussianBlur(delta, (19, 19), 3, borderType=cv2.BORDER_REFLECT_101)
        # Match the boundary's low-frequency tone while keeping the chosen
        # interior/reference detail. No change outside actual coverage.
        target = reference+low*(1-alpha[..., None])
    result = base+alpha[..., None]*(target-base)
    return result.astype(np.float32), alpha.astype(np.float32)
