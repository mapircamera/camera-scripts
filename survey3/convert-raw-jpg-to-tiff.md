---
description: Convert the Survey3 RAW+JPG pair of images into TIFF images
---

# Convert RAW+JPG to TIFF

{% hint style="danger" %}
The blank spaces in the `.bat` file are very important. Notice the spaces
between each argument in the main processing line. If the code format does not
match our examples it will not process correctly.
{% endhint %}

## Parameters

The parameters are processing options you can choose to use as follows:

| Parameter Name | Parameter Command | Explanation |
| --- | --- | --- |
| White Balance | `-wb` | Applies a white balance to the images (if supported) |

To apply a parameter it would look like:

```
python "Convert_Survey3_RAW_to_Tiff.py" -wb "input folder path" "output folder path"
```

If an image in your input folder does not support a parameter it will tell you
in the processing log window.

## Input and output folders

The input and output folder paths need to contain the path to your input image
folder and the location you want the processed images to be output to.

We typically create two local folders named **inFolder** and **outFolder**. The
inFolder is where you put the images you want to process and the outFolder will
contain the processed images.

Here is an example of the input and output folder paths:

```
python "Convert_Survey3_RAW_to_Tiff.py" "%~dp0\inFolder" "%~dp0\outFolder"
```

And with a parameter applied:

```
python "Convert_Survey3_RAW_to_Tiff.py" -wb "%~dp0\inFolder" "%~dp0\outFolder"
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
python "Convert_Survey3_RAW_to_Tiff.py" "%~dp0\inFolder" "%~dp0\outFolder"
pause >nul
```

When ran this script will open a command window and begin processing the images
in local folder inFolder and save the processed results to local folder
outFolder. The images will be corrected for white balance if supported on the
particular camera model.

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
