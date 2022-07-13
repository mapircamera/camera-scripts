# -*- coding: utf-8 -*-
"""
@author: MAPIR
"""

import sys, os
import glob
import cv2
import subprocess
import numpy as np
import rasterio
from ExifUtils import *


if sys.platform == "win32":
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW

modpath = os.path.dirname(os.path.realpath(__file__))

#returns all jpg files in the current directory
def get_jpg_files_in_dir(dir_name):
    file_paths = []
    file_paths.extend(glob.glob(dir_name + os.sep + "*.[jJ][pP][gG]"))
    return file_paths

#returns all tif files in the current directory
def get_tif_files_in_dir(dir_name):
    file_paths = []
    file_paths.extend(glob.glob(dir_name + os.sep + "*.[tT][iI][fF]"))
    return file_paths

def ApplyVig(infolder, infiles, ifTIFF, numFiles, outfolder, vigImg, dc):
    
    # Creating the folder to store the processed photos
    folderNum = 1
    path = outfolder + os.sep + 'Processed_'
    while(os.path.exists(path + str(folderNum))):
        folderNum += 1
    os.mkdir(path + str(folderNum))     
    outfolder = path + str(folderNum)
    
    if numFiles >= 1:
        counter = 0
        for file in infiles:    
            full_file_name = file.split('\\')[-1]
            print(f'\nProcessing {counter+1}/{numFiles}: {full_file_name}\n', end='')
            
            outputfilename = outfolder + os.sep + full_file_name
            inputfilename = infolder + os.sep + full_file_name
            
            img = cv2.imread(inputfilename,-1)
            
            #Create the dark current image (subMatrix) to be subtracted from img
            
            subMatrixB = np.full(shape = (3000,4000),fill_value = dc[0],dtype=int)
            subMatrixG = np.full(shape = (3000,4000),fill_value = dc[1],dtype=int)
            subMatrixR = np.full(shape = (3000,4000),fill_value = dc[2],dtype=int)

            if ifTIFF:
                subMatrixB = subMatrixB.astype("uint16")
                subMatrixG = subMatrixG.astype("uint16") 
                subMatrixR = subMatrixR.astype("uint16") 
            else:
                subMatrixB = subMatrixB.astype("uint8")
                subMatrixG = subMatrixG.astype("uint8") 
                subMatrixR = subMatrixR.astype("uint8")

            #Split img into 3 channels
            b,g,r = cv2.split(img)

            #Subtract dark current image from each channel of img
            b -= subMatrixB
            g -= subMatrixG
            r -= subMatrixR
            
            #Split vigImg into 3 channels
            vigB = rasterio.open(vigImg[0])
            vigG = rasterio.open(vigImg[1])
            vigR = rasterio.open(vigImg[2])
            
            vigB = vigB.read(1)
            vigG = vigG.read(1)
            vigR = vigR.read(1)
            
            #Apply flat field (vignette) correction by dividing vigImg and img per channel

            b = np.divide(b,vigB)
            g = np.divide(g,vigG)
            r = np.divide(r,vigR)
            
            #Clip off any values outside the bitdepth range (keep exposure lower to reduce clipping)
            if ifTIFF:
                b[b > 65535.0] = 65535.0
                b[b < 0.0] = 0.0
                
                g[g > 65535.0] = 65535.0
                g[g < 0.0] = 0.0
                
                r[r > 65535.0] = 65535.0
                r[r< 0.0] = 0.0
            
                color = cv2.merge((b,g,r))
                color = color.astype("uint16")
            else:
                b[b > 255.0] = 255.0
                b[b < 0.0] = 0.0
                
                g[g > 255.0] = 255.0
                g[g < 0.0] = 0.0
                
                r[r > 255.0] = 255.0
                r[r< 0.0] = 0.0
            
                color = cv2.merge((b,g,r))                
                color = color.astype("uint8") 

            #Save corrected img
            cv2.imwrite(outputfilename, color)
            ExifUtils.copy_simple(inputfilename,outputfilename,si)   
               
            counter += 1
            
    else:
        sys.exit("No files to process")
    
    print('Processing COMPLETE')
 
      
def main():

    inV = None
    ifTIFF = True

    if len(sys.argv) > 1:

        inf = sys.argv[1]
        outf = sys.argv[2]
        inV = sys.argv[3]

        if not os.path.exists(inf):
            sys.exit("Input file path does not exist or improperly formatted.")
        if not os.path.exists(outf):
            sys.exit("Output file path does not exist or improperly formatted.")
        
        # Pulling the photos to process    
        os.chdir(inf)
        infiles = []
        infiles.extend(get_tif_files_in_dir('.'))
        infiles.sort()
        numFiles = len(infiles)

        if numFiles == 0:
            ifTIFF = False
            infiles.extend(get_jpg_files_in_dir('.'))            
            numFiles = len(infiles)

        #Setting dark current values
        if ifTIFF:
            dc = [120, 119, 119]
        else:
            dc = [0, 0, 0]

    #Read the per-band flat field images into a single 3-band VigImg
    os.chdir(inV)
    vigImg = []
    vigImg.extend(get_tif_files_in_dir('.'))
    vigImg.sort()  
    
    #Call main function to apply vignette correction
    ApplyVig(inf, infiles, ifTIFF, numFiles, outf, vigImg, dc)
    
    
if __name__ == "__main__":
    main()
