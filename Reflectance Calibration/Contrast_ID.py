# -*- coding: utf-8 -*-
"""
@author: MAPIR, Inc
"""

import collections
import cv2
import cv2.aruco as aruco
from scipy import stats

#def is_grayscale_image(img):
    #return len(img.shape) == 2

def is_color_image(img):
    return len(img.shape) > 2

#def show_detected_targets(target_img, corners, ids):
    #draw_image = target_img.copy()
    #aruco.drawDetectedMarkers(draw_image, corners, ids)
    #resize = cv2.resize(draw_image, (1600, 1200), interpolation=cv2.INTER_LINEAR)
    #cv2.imshow('frame', resize)

def filter_detected_targets_by_id(corners, ids, target_id):
    return [i for i, j in zip(corners, ids) if j == target_id]

#def print_2d_list_frequencies(size, in_list):
    #frequencies = collections.Counter([])
    #for i in range(size):
        #row = in_list[i]
        #row_freqs = collections.Counter(row)
        #frequencies += row_freqs
        # print('Row ' + str(i) + ': ' + str(row_freqs))
    #print('Frequencies: ' + str(frequencies))

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