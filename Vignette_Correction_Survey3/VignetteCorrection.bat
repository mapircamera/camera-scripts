@echo on
call "C:\Users\MAPIR\Anaconda3\Scripts\activate.bat"
python "Vignette_Correction.py" "%~dp0\inFolder" "%~dp0\outFolder" "%~dp0\correctionFolder"
pause >nul