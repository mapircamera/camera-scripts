# -*- coding: utf-8 -*-
"""
@author: MAPIR, Inc
"""

import os, sys
import cv2
import numpy as np
import subprocess
from Calibration_Utils import get_calibration_coefficients_from_target_image
from ExifUtils import *

if sys.platform == "win32":
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW

def calibrate_channel(mult_values, value):
    slope = mult_values["slope"]
    intercept = mult_values["intercept"]
    return (slope * value) + intercept

def contrast_stretch_channel(global_cal_max, global_cal_min, rangeMax, value):
    value = rangeMax * ((value - global_cal_min) / (global_cal_max - global_cal_min))    
    return value

def calibrate_extrema(img):
    max_mins = {"redmaxs": [], "redmins": [], "greenmaxs": [], "greenmins": [], "bluemaxs": [], "bluemins": []}

    max_mins["redmaxs"].append(img[:, :, 2].max())
    max_mins["redmins"].append(img[:, :, 2].min())

    max_mins["greenmaxs"].append(img[:, :, 1].max())
    max_mins["greenmins"].append(img[:, :, 1].min())

    max_mins["bluemaxs"].append(img[:, :, 0].max())
    max_mins["bluemins"].append(img[:, :, 0].min())


    global_maxes = {"red": max(max_mins["redmaxs"]),
                    "green": max(max_mins["greenmaxs"]),
                    "blue": max(max_mins["bluemaxs"])}

    global_mins = {"red": min(max_mins["redmins"]),
                    "green": min(max_mins["greenmins"]),
                    "blue": min(max_mins["bluemins"])}

    extrema = {"calib": {"max": global_maxes, "min": global_mins}}
    return extrema 

def get_channel_extrema_for_image(image):
    channels = [
        image[:, :, 0],
        image[:, :, 1],
        image[:, :, 2]
    ]

    maxes = list(map(np.max, channels))
    mins = list(map(np.min, channels))
    return maxes, mins

def get_channel_extrema_for_project(inFolder):
    max_int = sys.maxsize
    min_int = -max_int - 1

    maxes = [min_int, min_int, min_int]
    mins = [max_int, max_int, max_int]

    for path, subdirs, files, in os.walk(inFolder):
        if files:
            for file_path in files:
                image = cv2.imread(os.path.join(path, file_path))
                tile_maxes, tile_mins = get_channel_extrema_for_image(image)

                for i in range(3):
                    if tile_maxes[i] > maxes[i]:
                        maxes[i] = int(tile_maxes[i])
                    if tile_mins[i] < mins[i]:
                        mins[i] = int(tile_mins[i])

    return maxes, mins

def get_global_calib_extrema(calibration_values, global_max, global_min):
    global_cal_maxes = []
    global_cal_mins = []

    global_cal_maxes.append( int(calibrate_channel(calibration_values["red"], global_max[0])) )
    global_cal_mins.append( int(calibrate_channel(calibration_values["red"], global_min[0])) )

    global_cal_maxes.append( int(calibrate_channel(calibration_values["green"], global_max[1])) )
    global_cal_mins.append( int(calibrate_channel(calibration_values["green"], global_min[1])) )
    
    global_cal_maxes.append( int(calibrate_channel(calibration_values["blue"], global_max[2])) )
    global_cal_mins.append( int(calibrate_channel(calibration_values["blue"], global_min[2])) )
    
    global_cal_max = max(global_cal_maxes)
    global_cal_min = min(global_cal_mins)

    return global_cal_max, global_cal_min


def main():
    
    if len(sys.argv) > 1:
         calib_photo = sys.argv[1]
         inFolder = sys.argv[2]
         outFolder = sys.argv[3]

    print('\n(1/3) Computing Calibration Values')
    calibration_values, FileType_calib = get_calibration_coefficients_from_target_image(calib_photo, inFolder)

    print('\n(2/3) Analyzing Input Images')
    maxes, mins  = get_channel_extrema_for_project(inFolder)
    global_cal_max, global_cal_min = get_global_calib_extrema(calibration_values, maxes, mins)

    print('\n(3/3) Calibrating Images\n')
    
    for path, subdirs, files, in os.walk(inFolder):
        if files:
            for file_name in files:
                img = cv2.imread(os.path.join(path, file_name))
                
                red = img[:, :, 2]
                green = img[:, :, 1]
                blue = img[:, :, 0]                

                red = calibrate_channel(calibration_values["red"], red)
                green = calibrate_channel(calibration_values["green"], green)
                blue = calibrate_channel(calibration_values["blue"], blue)

                full_path_in = os.path.join(inFolder, file_name)
                full_path_out = os.path.join(outFolder, file_name)

                if FileType_calib == "JPG":
                    red = contrast_stretch_channel(global_cal_max, global_cal_min, 255.0, red)
                    green = contrast_stretch_channel(global_cal_max, global_cal_min, 255.0, green)
                    blue= contrast_stretch_channel(global_cal_max, global_cal_min, 255.0, blue)

                    imgJPG = cv2.merge((blue, green, red))
                    imgJPG = imgJPG.astype("uint8")
                    cv2.imencode(".jpg", imgJPG)
                    cv2.imwrite(full_path_out, imgJPG) 
                    print(full_path_out)
                    

                elif FileType_calib == "TIFF":
                    
                    red = contrast_stretch_channel(global_cal_max, global_cal_min, 65535.0, red)
                    green = contrast_stretch_channel(global_cal_max, global_cal_min, 65535.0, green)
                    blue= contrast_stretch_channel(global_cal_max, global_cal_min, 65535.0, blue)
                  
                    imgTIF = cv2.merge((blue, green, red))
                    imgTIF = imgTIF.astype("uint32")
                    imgTIF = imgTIF.astype("uint16")
                    cv2.imencode(".tif", imgTIF)
                    cv2.imwrite(full_path_out, imgTIF)
                    ExifUtils.copy_simple(full_path_in,full_path_out,si)
                    
                    print(full_path_out)

    print('\nFinished Processing\n') 
    
if __name__ == '__main__':
    main()