import numpy as np
from scipy import ndimage


def rotate(array, angle):
    return ndimage.rotate(input=array, angle=angle, order=0, reshape=False)


def flip_h(array):
    return np.fliplr(array)


def flip_v(array):
    return np.flipud(array)


def do_nothing(array):
    return array


def flip_and_rotate(array, select, angle):
    """Geometric augmentation"""
    # select flipping or non-flipping mode (randomly)
    functions = [flip_h, flip_v, do_nothing]
    arr = functions[select[0]](array)
    return rotate(arr, angle=angle)
