# -*- coding: utf-8 -*-
"""
@author: MAPIR, Inc
"""

from scipy import stats

def is_color_image(img):
    return len(img.shape) > 2

def filter_detected_targets_by_id(corners, ids, target_id):
    return [i for i, j in zip(corners, ids) if j == target_id]

def contrast_stretch(img, threshold):
    bit_depth = int(img.dtype.name[4:])
    pixel_range_max = 2**bit_depth-1
    for pixel_row in img:
        pixel_row[pixel_row <= threshold] = 0 # Black
        pixel_row[pixel_row > threshold] = pixel_range_max # White

def midpoint_threshold_contrast_stretch(img):
    stretch_img = img.copy()
    threshold = (stretch_img.max() + 1 - stretch_img.min()) / 2 - 1
    contrast_stretch(stretch_img, threshold)
    return stretch_img

def mode_threshold_contrast_stretch(img):
    stretch_img = img.copy()
    threshold = stats.mode(stretch_img.flatten())[0][0]
    contrast_stretch(stretch_img, threshold)
    return stretch_img