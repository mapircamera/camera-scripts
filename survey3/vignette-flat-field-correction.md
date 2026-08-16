---
description: Correct the vignette of the Survey3 images
---

# Vignette (Flat Field) Correction

```
python "Vignette_Correction.py" "input folder path" "output folder path" "flat field correction images path"
```

{% hint style="danger" %}
The blank spaces in the `.bat` file are very important. Notice the spaces
between each argument in the main processing line. If the code format does not
match our examples it will not process correctly.
{% endhint %}

## Input and output folders

The input and output folder paths need to contain the path to your input image
folder and the location you want the processed images to be output to.

We typically create two local folders named **inFolder** and **outFolder**. The
inFolder is where you put the images you want to process and the outFolder will
contain the processed images.

## The correction folder

The **correctionFolder** must contain 3 images titled:

```
Average Flat Field Correction ImageB.tif
Average Flat Field Correction ImageG.tif
Average Flat Field Correction ImageR.tif
```

Here is an example of the input and output folder paths:

```
python "Vignette_Correction.py" "%~dp0\inFolder" "%~dp0\outFolder" "%~dp0\correctionFolder"
```

The images in the output folder will be saved to a subfolder named
**Processed\_N**, where N starts at 1 and counts up depending on if there are
already other Processed folders located in the output folder or not. This is
ideal so that you can output to the same location every time and it will not
overwrite your previously processed images.

## The .bat file

Here is an example of an entire `.bat` ready to be run:

```batch
@echo on
call "C:\Users\MAPIR\Anaconda3\Scripts\activate.bat"
python "Vignette_Correction.py" "%~dp0\inFolder" "%~dp0\outFolder" "%~dp0\correctionFolder"
pause >nul
```

When ran this script will open a command window and begin processing the images
in local folder inFolder and save the processed results to local folder
outFolder. The images will be corrected for vignette using the correction images
in the correctionFolder.

Once you have made all the necessary changes to the `.bat` file you can save it
to begin processing.

## Running the Script

To run the script simply double-click or right-click and select **Open** on the
`.bat` file.

A command window will pop up and provide feedback during the processing.

If you need to stop the processing you can press **Ctrl+C** and then answer the
prompt with **Y+Enter**.

The script is done processing when you see the final text **Processing
COMPLETE**.
