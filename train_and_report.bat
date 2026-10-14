@echo off
setlocal
pushd "%~dp0"
set "PYTHONUNBUFFERED=1"
set "PYTHONUTF8=1"
if defined MBDNN_PYTHON (
    set "mbdnn_python=%MBDNN_PYTHON%"
) else if exist ".venv\Scripts\python.exe" (
    set "mbdnn_python=%~dp0.venv\Scripts\python.exe"
) else (
    echo Python environment not found. Create .venv or set MBDNN_PYTHON.
    popd
    pause
    exit /b 1
)
"%mbdnn_python%" -u -m mbdnn.automation %*
set "mbdnn_exit=%ERRORLEVEL%"
echo.
if "%mbdnn_exit%"=="0" (
    echo Workflow completed. See the report or check log path printed above.
) else (
    echo Workflow failed or interrupted. Check the log. Run again to resume completed models.
)
popd
if /i not "%MBDNN_NO_PAUSE%"=="1" pause
exit /b %mbdnn_exit%
