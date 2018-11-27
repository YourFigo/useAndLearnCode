REM @echo off

@echo off


set name=FileList.txt

set time=2017060309460

set exc=.mp3

for /l %%i in (7001,1,7100) do (
echo %time%%%i%exc% >>%name%
)