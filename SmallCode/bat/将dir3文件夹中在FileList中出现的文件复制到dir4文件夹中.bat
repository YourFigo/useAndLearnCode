REM @echo off
@echo off
set str = "E:\GitCode\useAndLearnCode\SmallCode\bat\"
set str1="dir3"
set str2="dir4"
for /f %%i in (FileList.txt) do (echo F|(xcopy %str%%str1%\%%i %str%%str2%\%%i /y))
pause