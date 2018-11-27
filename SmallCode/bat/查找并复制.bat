REM @echo off
@echo off
set str="D:\dir1\20170603094607000.mp3"
set str1="D:\dir3"
for /f %%i in (FileList.txt) do (echo F|(xcopy %str% %str1%\%%i /y))
pause