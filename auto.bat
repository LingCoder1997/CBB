@echo off
setlocal

set SDK_ROOT=D:\Pycharm_workplace\CBB

cd %SDK_ROOT%

python setup.py bdist_wheel

for /f "tokens=*" %%i in ('dir /b /o:-d %SDK_ROOT%\dist\*.whl') do (
    set LATEST_WHEEL=%%i
    goto :found
)
:found

python -m pip install %SDK_ROOT%\dist\%LATEST_WHEEL% --force-reinstall

endlocal