@echo off
setlocal
pushd "%~dp0"
if defined MBDNN_PYTHON (
    set "mbdnn_python=%MBDNN_PYTHON%"
) else if exist ".venv\Scripts\python.exe" (
    set "mbdnn_python=.venv\Scripts\python.exe"
) else (
    echo Create .venv first, or invoke an installed Python with -m mbdnn.
    popd
    exit /b 1
)
if "%~1"=="" (
    "%mbdnn_python%" -m mbdnn plan
) else (
    "%mbdnn_python%" -m mbdnn %*
)
set "mbdnn_exit=%ERRORLEVEL%"
popd
exit /b %mbdnn_exit%
