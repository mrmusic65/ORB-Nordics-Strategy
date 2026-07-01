@echo off
setlocal enabledelayedexpansion

if "%~1"=="" (
    echo Usage: run_all.bat ^<folder_with_csv_files^>
    exit /b 1
)

set "CSV_DIR=%~1"
set "SCRIPT_DIR=%~dp0"
set "SUMMARY=%SCRIPT_DIR%output\pipeline_summary.csv"

if not exist "%CSV_DIR%" (
    echo Folder does not exist: %CSV_DIR%
    exit /b 1
)

if exist "%SUMMARY%" del "%SUMMARY%"

for %%F in ("%CSV_DIR%\*.csv") do (
    echo Running %%~nF
    py "%SCRIPT_DIR%orb_features_pipeline.py" "%%~fF" "%%~nF"
    if errorlevel 1 (
        echo Pipeline failed for %%~nF
        exit /b 1
    )
)

echo Done. Summary: %SUMMARY%
