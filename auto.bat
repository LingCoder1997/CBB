@echo off
setlocal

set SDK_ROOT=D:\Pycharm_workplace\CBB

if not defined CONDA_PREFIX (
    echo Please activate your conda environment before running this script.
    exit /b 1
)

cd %SDK_ROOT%

python setup.py bdist_wheel

for /f "tokens=*" %%i in ('dir /b /o:-d %SDK_ROOT%\dist\*.whl') do (
    set LATEST_WHEEL=%%i
    goto :found
)
:found

python -m pip install %SDK_ROOT%\dist\%LATEST_WHEEL% --force-reinstall

endlocal