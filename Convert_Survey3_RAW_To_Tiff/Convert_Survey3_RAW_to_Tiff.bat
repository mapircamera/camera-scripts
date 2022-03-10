@echo on
call "C:\Users\MAPIR\Anaconda3\Scripts\activate.bat"
python "Convert_Survey3_RAW_to_Tiff.py" "%~dp0\inFolder" "%~dp0\outFolder"
pause >nul