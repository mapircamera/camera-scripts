@echo on
call "C:\Users\MAPIR\Anaconda3\Scripts\activate.bat"
python "calibration.py" "%~dp0calib\calib.jpg" "%~dp0inFolder" "%~dp0outFolder"
pause >nul 