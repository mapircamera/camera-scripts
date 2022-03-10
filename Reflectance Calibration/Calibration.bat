@echo on
call "C:\Users\MAPIR\Anaconda3\Scripts\activate.bat"
python "calibration.py" "%~dp0\calib\calib.tif" "%~dp0\inFolder" "%~dp0\outFolder"
pause >nul 