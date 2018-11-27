REM @echo off
@echo off
set str1="D:\dir3"
set str2="D:\dir4"
for /f %%i in (FileList.txt) do (echo F|(xcopy %str1%\%%i %str2%\%%i /y))
pause