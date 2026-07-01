@echo off
setlocal

if "%~1"=="" (
    echo Usage: run_walkforward.bat ^<feature_folder^> ^<intraday_folder^>
    exit /b 1
)

if "%~2"=="" (
    echo Usage: run_walkforward.bat ^<feature_folder^> ^<intraday_folder^>
    exit /b 1
)

set "FEATURE_DIR=%~1"
set "INTRADAY_DIR=%~2"
set "SCRIPT_DIR=%~dp0"

call :run_period 2009-01-01 2011-12-31
if errorlevel 1 exit /b 1
call :run_period 2012-01-01 2014-12-31
if errorlevel 1 exit /b 1
call :run_period 2015-01-01 2017-12-31
if errorlevel 1 exit /b 1
call :run_period 2018-01-01 2020-12-31
if errorlevel 1 exit /b 1
call :run_period 2021-01-01 2023-12-31
if errorlevel 1 exit /b 1

echo Walk-forward batch complete.
exit /b 0

:run_period
set "PERIOD_START=%~1"
set "PERIOD_END=%~2"

for %%M in (0.5 1.0 1.5) do (
    echo Running ATR %%M from %PERIOD_START% to %PERIOD_END%
    py "%SCRIPT_DIR%orb_backtest.py" "%FEATURE_DIR%" "%INTRADAY_DIR%" %%M %PERIOD_START% %PERIOD_END%
    if errorlevel 1 (
        echo Walk-forward run failed: ATR %%M from %PERIOD_START% to %PERIOD_END%
        exit /b 1
    )
)

exit /b 0
