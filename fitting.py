"""Robust multiscale parameter fitting with a held-out review image."""
import copy
import cv2
import numpy as np
from .color import defaults, render, LUMA, LIMITS
from .common import resize, interrupted, resize_to


def correspondence(a, b, valid):
    ya, yb = a @ LUMA, b @ LUMA
    ax = cv2.Sobel(ya, cv2.CV_32F, 1, 0, ksize=3)/8
    ay = cv2.Sobel(ya, cv2.CV_32F, 0, 1, ksize=3)/8
    bx = cv2.Sobel(yb, cv2.CV_32F, 1, 0, ksize=3)/8
    by = cv2.Sobel(yb, cv2.CV_32F, 0, 1, ksize=3)/8
    an, bn = np.hypot(ax, ay), np.hypot(bx, by)
    cosine = (ax*bx+ay*by)/(an*bn+1e-8)
    strong = np.maximum(an, bn) > .025
    bad = strong & ((np.minimum(an, bn) < .003) | (cosine < .35))
    weight = np.where(bad, .05, np.where(strong & (cosine < .75), .35, 1)).astype(np.float32)
    # Coherent structural disagreement is excluded rather than chased by color.
    mismatch = cv2.GaussianBlur(bad.astype(np.float32), (0, 0), 2)
    weight *= 1-np.clip((mismatch-.2)/.5, 0, 1)
    weight *= np.where(b.min(-1) > .988, .15, 1)*valid
    return weight.astype(np.float32), mismatch


def loss(out, target, weights, holdout=False, enhance=False):
    selected = (np.arange(weights.size).reshape(weights.shape) % 7 == 0) == holdout
    w = weights*selected
    d = np.clip(out, 0, 1)-target
    def huber(x):
        x = np.abs(x)
        return np.where(x < .06, x*x, .12*x-.0036)
    neutral = 1-np.clip(np.ptp(target, axis=-1)/.16, 0, 1)
    err = huber(d).mean(-1)+neutral*(huber(d[..., 0]-d[..., 1])+huber(d[..., 2]-d[..., 1]))
    if enhance:
        err += 2*huber(d @ LUMA)+.5*huber(np.ptp(out, axis=-1)-np.ptp(target, axis=-1))
        err += 3*np.square(np.maximum(0, out-np.maximum(.97, target+.02))).mean(-1)
        err += 3*np.square(np.maximum(0, np.minimum(.03, target-.02)-out)).mean(-1)
    return float((err*w).sum()/max(float(w.sum()), 1))


def penalty(p):
    d = np.diff(p['curves'], axis=1)*4
    return float(.000015*np.square(np.diff(d, axis=1)).sum()
                 +.000025*np.square(p['hsl']/40).sum()
                 +.000015*np.square((p['hsl']-np.roll(p['hsl'], 1, axis=0))/40).sum())


def descriptors(p, family, scale):
    if family == 'basic':
        return [('basic', (i,), *LIMITS[i], (.55 if i == 0 else 14)*scale) for i in range(10)]
    if family == 'tone':
        return [('curves', (0, i), 0, 1, .05*scale) for i in range(1, 4)]
    if family == 'channels':
        return [('curves', (c, i), 0, 1, .025*scale) for c in range(1, 4) for i in range(1, 4)]
    return [('hsl', (c, k), -40 if k == 0 else -70, 40 if k == 0 else 70, 10*scale)
            for c in range(8) for k in range(3)]


def fit_parameters(a, b, valid, enhance=False):
    p = defaults()
    if np.max(np.abs(a-b)) < 1e-7:
        return p
    check = resize(a, 384); goal = resize_to(b, check.shape[:2])
    vw = resize_to(valid, check.shape[:2]); cw, _ = correspondence(check, goal, vw)
    def review(q):
        return loss(render(check, q), goal, cw, True, enhance)
    best_score, best_p = review(p), copy.deepcopy(p)
    for stage, edge, passes in [(0, 96, 6), (1, 192, 4), (2, 192, 4)]:
        x = resize(a, edge); y = resize_to(b, x.shape[:2]); v = resize_to(valid, x.shape[:2])
        weights, _ = correspondence(x, y, v)
        def evaluate():
            return loss(render(x, p), y, weights, enhance=enhance)+penalty(p)
        if stage == 0:
            score = evaluate(); exp = 0
            for e in [-2, -1, 1, 2, 3]:
                p['basic'][0] = e; val = evaluate()
                if val < score: score, exp = val, e
            p['basic'][0] = exp
        for family in ['basic', 'tone', 'channels', 'hsl']:
            ds = descriptors(p, family, .4 if stage else 1)
            best = evaluate()
            for run in range(passes):
                interrupted()
                for key, index, lower, upper, step in ds:
                    old = p[key][index]; chosen = old
                    if key == 'curves':
                        c, i = index
                        lower = p[key][c, i-1]+(.01 if c == 0 else 0)
                        upper = p[key][c, i+1]-(.01 if c == 0 else 0)
                    for direction in [-1, 1]:
                        p[key][index] = np.clip(old+direction*step*2**(-run), lower, upper)
                        score = evaluate()
                        if score < best-1e-10: best, chosen = score, p[key][index]
                    p[key][index] = chosen
            score = review(p)
            if score < best_score:
                best_score, best_p = score, copy.deepcopy(p)
            else:
                p = copy.deepcopy(best_p)
    return p
