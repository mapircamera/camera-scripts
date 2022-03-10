# -*- coding: utf-8 -*-
"""
@author: MAPIR, Inc
"""
import os
import sys
import glob
import numpy as np
import copy
import cv2
import subprocess
from PIL import Image
from ExifUtils import *
import exifread
import warnings


if sys.platform == "win32":
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    
warnings.filterwarnings("ignore", category=DeprecationWarning)

COLOR_CORRECTION_VECTORS = [1.398822546, -0.09047482163, 0.1619316638, -0.01290435996, 0.8994362354, 0.1134681329, 0.007306902204, -0.05995989591, 1.577814579]#101018

MAKE = 'MAPIR'
MODELS = ['Survey3W_RGB',
          'Survey3W_OCN',
          'Survey3W_RGN',
          'Survey3W_NGB',
          'Survey3W_RE',
          'Survey3W_NIR',
          'Survey3N_RGB',
          'Survey3N_OCN',
          'Survey3N_RGN',
          'Survey3N_NGB',
          'Survey3N_RE',
          'Survey3N_NIR'
          ]
    
def Survey3_Convert_RAW_to_Tiff(infolder, outfolder, wb):
    
    # Creating the folder to store the processed photos
    folderNum = 1
    path = outfolder + os.sep + 'Processed_'  #iterate output folder
    while(os.path.exists(path + str(folderNum))):
        folderNum += 1
    os.mkdir(path + str(folderNum))     
    outfolder = path + str(folderNum)
    
    # Pulling the photos to process    
    os.chdir(infolder)
    infiles = []
    infiles.extend(get_raw_files_in_dir('.'))
    infiles.extend(get_jpg_files_in_dir('.'))
    infiles.sort()
    numFiles = len(infiles)
    
    # Check if there's at least 2 files in the folder, else quit
    if numFiles > 1:
        first_files_jpg = "JPG" in infiles[0].upper() and "JPG" in infiles[1].upper()       # returns true if first two files are JPG
        first_files_rawjpg = "RAW" in infiles[0].upper() and "JPG" in infiles[1].upper()    # returns true if first two files are RAW and JPG
        if first_files_jpg or first_files_rawjpg:
            if first_files_rawjpg:
                if numFiles % 2 != 0:
                    sys.exit("There is not an equal number of JPG and RAW images.")
            fil = check_make_model(infiles, infolder)
            
            if first_files_rawjpg:
                files_to_process = infiles[::2]
            else:
                files_to_process = infiles
            count = 1
    
            # Convert each photo to tif and copy over exif data
            for file in files_to_process:
                if first_files_rawjpg:
                    try:
                        print(f'Processing file {count}/{len(files_to_process)} {file[2:]}')
                        # Pull pixel data from RAW image and format as 36000000 by 4 matrix
                        data = np.fromfile(file, dtype=np.uint8)
                        data = np.unpackbits(data)
                        datsize = data.shape[0]
                        data = data.reshape((int(datsize / 4), 4))
        
                        # Switch even rows and odd rows
                        temp = copy.deepcopy(data[0::2])
                        temp2 = copy.deepcopy(data[1::2])
                        data[0::2] = temp2
                        data[1::2] = temp
                        
                        # Repack into image file
                        udata = np.packbits(np.concatenate([data[0::3], np.array([0, 0, 0, 0] * 12000000, dtype=np.uint8).reshape(12000000,4),   data[2::3], data[1::3]], axis=1).reshape(192000000, 1)).tobytes()
                        img = np.fromstring(udata, np.dtype('u2'), (4000 * 3000)).reshape((3000, 4000))
                    except Exception as e:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        print(str(e) + ' Line: ' + str(exc_tb.tb_lineno))
                        
                    try:
                        color = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB).astype("float32")
                    except Exception as e:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        print(str(e) + ' Line: ' + str(exc_tb.tb_lineno))
                    
                    # White balance color correction 
                    if wb and fil[count-1] == 'RGB':
                        print('Applying White Balance Correction')
                        color = color / 65535.0
                        color = color_correction(color)
                        color = color * 65535.0
                        print('DONE')
                    elif wb and not (fil[count-1] == 'RGB'):
                        print('White Balance Correction Not Available')
                    
                    if not (wb and fil[count-1] == 'RGB'):
                        color = color * 65535.0
                    
                    color = color.astype("uint32")
                    color = color.astype("uint16") 
                    color[color>=65535] = 65535
                    if not (wb and fil[count-1] == 'RGB'):
                        color = cv2.bitwise_not(color)
                    filename = file.split('.')
                    outputfilename = filename[1] + '.tif'
                    cv2.imencode(".tif", color)                      
                
                cv2.imwrite(outfolder + outputfilename, color)
                file_index = count*2 -1 if first_files_rawjpg else count-1    
                ExifUtils.copy_simple(infiles[file_index],outfolder + outputfilename,si)
                count+=1 
                print()
        else:
            sys.exit("Incorrect file structure. Please arrange files in a RAW, JPG, RAW, JPG format.")
    else:
        sys.exit("No images to process")
    
    print("Processing COMPLETE") 
    
#returns all raw files in the current directory
def get_raw_files_in_dir(dir_name):
    return glob.glob(dir_name + os.sep + "*.[rR][aA][wW]")

#returns all jpg files in the current directory
def get_jpg_files_in_dir(dir_name):
    file_paths = []
    file_paths.extend(glob.glob(dir_name + os.sep + "*.[jJ][pP][gG]"))
    file_paths.extend(glob.glob(dir_name + os.sep + "*.[jJ][pP][eE][gG]"))
    return file_paths

# Reapplies true color values to the pixel data if the -wb flag is set
def color_correction(color):
    roff = 0.0
    goff = 0.0
    boff = 0.0

    red_coeffs = COLOR_CORRECTION_VECTORS[6:9]
    green_coeffs = COLOR_CORRECTION_VECTORS[3:6]
    blue_coeffs = COLOR_CORRECTION_VECTORS[:3]

    color[:, :, 2] = (red_coeffs[0] * color[:, :, 0]) + (red_coeffs[1] * color[:, :, 1]) + (red_coeffs[2] * color[:, :, 2]) + roff
    color[:, :, 1] = (green_coeffs[0] * color[:, :, 0]) + (green_coeffs[1] * color[:, :, 1]) + (green_coeffs[2] * color[:, :, 2]) + goff
    color[:, :, 0] = (blue_coeffs[0] * color[:, :, 0]) + (blue_coeffs[1] * color[:, :, 1]) + (blue_coeffs[2] * color[:, :, 2]) + boff

    color[color > 1.0] = 1.0
    color[color < 0.0] = 0.0

    return color

# Checks if all the image files in folder are the proper MAKE/MODEL
# raw_jpg is a boolean describing the presense of both raw and jpg's being in the folder.
# Also returns a list of the filter type from model for each image to check if RGB, RGN etc...
# If the wrong MAKE or MODEL is found, all exif tags and data are printed and system exits
# The imagename is printed on sys.exit
def check_make_model(files, folder, raw_jpg=True):    
    if raw_jpg:
        files = files[1::2]
    
    filterVals = []
    counter = 0
    for imagename in files:
        image = Image.open(folder + os.sep + imagename[2:])
        exifdata = image.getexif()
        makeData = exifdata.get(271,271)
        modelData = exifdata.get(272,272)
        filterVals.append(modelData[-3:])
        counter += 1
        if makeData != MAKE or modelData not in MODELS:
            exitString = imagename + " Make/Model not valid"
            myExif = open(folder + os.sep + imagename[2:], 'rb')
            tags = exifread.process_file(myExif)    
            for tag in tags.keys():
                print (f"Key: {tag}, value {tags[tag]}")
            
            # If you have a photo with corrupted exif
            # but you still want to use it, just comment sys.exit(exitString)
            #print(exitString)
            myExif.close()
            sys.exit(exitString) 
            
    return filterVals   

def main():
    wb = False
    if len(sys.argv) > 1:
        if len(sys.argv) == 3:
            inf = sys.argv[1]
            outf = sys.argv[2]
        elif len(sys.argv) == 4:
            if sys.argv[1] == '-wb':
                wb = True
            inf = sys.argv[2]
            outf = sys.argv[3]
        else:
            sys.exit("Improper formatting of commandline arguments.")
       
    if not os.path.exists(inf):
        sys.exit("Input file path does not exist or improperly formatted.")
    if not os.path.exists(outf):
        sys.exit("Output file path does not exist or improperly formatted.")

    print()
    Survey3_Convert_RAW_to_Tiff(inf, outf, wb)
    
if __name__ == "__main__":
    main()