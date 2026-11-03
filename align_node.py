"""Standalone registration in the original image coordinate system."""
import tempfile
import numpy as np
import torch
import cv2
from .nodes import estimate, interrupted


def warp_reference(reference, source, matrix):
    h, w = source.shape[:2]
    rh, rw = reference.shape[:2]
    # estimate() maps normalized source pixel centers into reference centers.
    n = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    to_norm = np.array([[1/w, 0, .5/w], [0, 1/h, .5/h], [0, 0, 1]])
    to_pixels = np.array([[rw, 0, -.5], [0, rh, -.5], [0, 0, 1]])
    m = to_pixels @ n @ to_norm
    result = np.empty_like(source)
    coverage = np.empty((h, w), np.float32)
    # Bounded coordinate buffers, even for large originals.
    for top in range(0, h, 256):
        interrupted()
        y, x = np.mgrid[top:min(h, top+256), :w]
        d = m[2, 0]*x + m[2, 1]*y + m[2, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            mx = (m[0, 0]*x+m[0, 1]*y+m[0, 2])/d
            my = (m[1, 0]*x+m[1, 1]*y+m[1, 2])/d
        valid = np.isfinite(mx)&np.isfinite(my)&(mx>=-1e-5)&(my>=-1e-5)&(mx<=rw-1+1e-5)&(my<=rh-1+1e-5)
        mx = np.nan_to_num(mx, nan=-1, posinf=-1, neginf=-1).astype(np.float32)
        my = np.nan_to_num(my, nan=-1, posinf=-1, neginf=-1).astype(np.float32)
        warped = cv2.remap(reference, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        result[top:top+len(y)] = np.where(valid[..., None], warped, source[top:top+len(y)])
        coverage[top:top+len(y)] = valid
    return result, coverage


class JayouSDRAWAlign:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image_a': ('IMAGE',), 'image_b': ('IMAGE',),
            '对齐方式': (['SIFT + MAGSAC++', 'LightGlue + MAGSAC++', '平移'],)}}
    RETURN_TYPES = ('IMAGE', 'IMAGE', 'MASK')
    RETURN_NAMES = ('原图', '对齐参考图', '有效重叠范围')
    FUNCTION = 'process'
    CATEGORY = '嘉游SDRAW'

    def process(self, image_a, image_b, 对齐方式='SIFT + MAGSAC++'):
        modes = {'SIFT + MAGSAC++':'sift', 'LightGlue + MAGSAC++':'lightglue', '平移':'translation'}
        if 对齐方式 not in modes:
            raise ValueError('无效对齐方式')
        for im in (image_a, image_b):
            if not isinstance(im, torch.Tensor) or im.ndim != 4 or im.shape[-1] != 3 or min(im.shape[:3]) < 1:
                raise ValueError('IMAGE 必须是 [B,H,W,3]')
            if im.shape[1]*im.shape[2] > 100_000_000 or max(im.shape[1:3]) >= 32767:
                raise ValueError('对齐输入超过尺寸限制：单图 1 亿像素、单边小于 32767')
            if not torch.isfinite(im).all() or im.min() < 0 or im.max() > 1:
                raise ValueError('IMAGE 需要有限的 0–1 RGB 值')
        if image_b.shape[0] not in (1, image_a.shape[0]):
            raise ValueError('参考批次数需为 1 或与原图一致')
        results, masks = [], []
        for i in range(image_a.shape[0]):
            interrupted()
            a = image_a[i].detach().cpu().float().numpy()
            b = image_b[0 if image_b.shape[0] == 1 else i].detach().cpu().float().numpy()
            with tempfile.TemporaryDirectory(prefix='jayou-align-') as tmp:
                transform = estimate(a, b, modes[对齐方式], tmp)
            result, valid = warp_reference(b, a, transform['matrix'])
            results.append(torch.from_numpy(result)); masks.append(torch.from_numpy(valid))
        return image_a, torch.stack(results), torch.stack(masks)
