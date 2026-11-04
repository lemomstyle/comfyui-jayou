"""Normalized pixel-center registration and bounded-memory warping."""
import cv2
import numpy as np
import tempfile
from pathlib import Path
from .common import resize, interrupted, chunks
from .color import LUMA
from .geometry import estimate_geometry


def estimate(a, b, mode):
    interrupted()
    if mode == '关闭':
        if abs(np.log((a.shape[1]/a.shape[0])/(b.shape[1]/b.shape[0]))) > .01:
            raise ValueError('关闭对齐时，两图宽高比必须一致')
        return np.eye(3)
    sa, sb = resize(a, 1280), resize(b, 1280)
    if mode == '平移':
        # Correlate gradients to reduce sensitivity to color/exposure differences.
        sb = cv2.resize(sb, (sa.shape[1], sa.shape[0]), interpolation=cv2.INTER_AREA)
        ga, gb = sa @ LUMA, sb @ LUMA
        ga = cv2.Laplacian(ga, cv2.CV_32F); gb = cv2.Laplacian(gb, cv2.CV_32F)
        window = cv2.createHanningWindow((sa.shape[1], sa.shape[0]), cv2.CV_32F)
        shift, confidence = cv2.phaseCorrelate(ga, gb, window)
        if not np.isfinite(shift).all() or confidence < .1:
            raise ValueError('平移对齐不可靠，请选择 SIFT 或检查图片')
        dx, dy = shift[0]/sa.shape[1], shift[1]/sa.shape[0]
        if abs(dx) > .3 or abs(dy) > .3:
            raise ValueError('平移量过大，无法可靠对齐')
        return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)
    if mode == 'LightGlue + MAGSAC++':
        from .geometry import align
        with tempfile.TemporaryDirectory(prefix='jayou-align-') as tmp:
            pa, pb = Path(tmp)/'a.png', Path(tmp)/'b.png'
            for p, im in [(pa, sa), (pb, sb)]:
                cv2.imwrite(str(p), cv2.cvtColor(np.rint(im*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            result = align(str(pa), str(pb))
    elif mode == 'SIFT + MAGSAC++':
        detector = cv2.SIFT_create(nfeatures=4096)
        features = [detector.detectAndCompute(np.rint((im @ LUMA)*255).astype(np.uint8), None) for im in (sa, sb)]
        (ka, da), (kb, db) = features
        if da is None or db is None:
            raise ValueError('图像纹理不足，无法对齐')
        pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .75*pair[1].distance]
        aa = np.float32([ka[m.queryIdx].pt for m in good])
        bb = np.float32([kb[m.trainIdx].pt for m in good])
        result = estimate_geometry(aa, bb, (sa.shape[1], sa.shape[0]), (sb.shape[1], sb.shape[0]))
    else:
        raise ValueError('未知对齐方式')
    interrupted()
    return np.asarray(result['matrix']).reshape(3, 3)


def warp_strip(reference, matrix, shape, top, bottom):
    h, w = shape; rh, rw = reference.shape[:2]
    y, x = np.mgrid[top:bottom, :w]; u = (x+.5)/w; v = (y+.5)/h
    d = matrix[2, 0]*u+matrix[2, 1]*v+matrix[2, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        mx = ((matrix[0, 0]*u+matrix[0, 1]*v+matrix[0, 2])/d*rw-.5)
        my = ((matrix[1, 0]*u+matrix[1, 1]*v+matrix[1, 2])/d*rh-.5)
    valid = np.isfinite(mx)&np.isfinite(my)&(mx >= -1e-5)&(my >= -1e-5)&(mx <= rw-1+1e-5)&(my <= rh-1+1e-5)
    mx = np.nan_to_num(mx, nan=-1, posinf=-1, neginf=-1).astype(np.float32)
    my = np.nan_to_num(my, nan=-1, posinf=-1, neginf=-1).astype(np.float32)
    out = cv2.remap(reference, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, valid.astype(np.float32)


def warp(reference, source, matrix):
    out = np.empty_like(source); valid = np.empty(source.shape[:2], np.float32)
    for top, bottom in chunks(*source.shape[:2]):
        b, m = warp_strip(reference, matrix, source.shape[:2], top, bottom)
        out[top:bottom] = np.where(m[..., None] > 0, b, source[top:bottom]); valid[top:bottom] = m
    return out, valid
