import cv2
import numpy as np


def interrupted():
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except ImportError:
        return
    throw_exception_if_processing_interrupted()


def resize(a, edge):
    h, w = a.shape[:2]; factor = min(1, edge/max(h, w))
    return cv2.resize(a, (max(1, round(w*factor)), max(1, round(h*factor))), interpolation=cv2.INTER_AREA)


def resize_to(a, shape):
    return cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def chunks(height, width, budget=262144):
    step = max(1, budget//width)
    for top in range(0, height, step):
        interrupted()
        yield top, min(height, top+step)
