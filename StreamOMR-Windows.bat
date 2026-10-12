@echo off
rem StreamOMR launcher for Windows - double-click in Explorer.
rem First run creates .venv and installs requirements.txt; later runs start the server directly.
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
set PYTHONUTF8=1

set "PY=.venv\Scripts\python.exe"
set "STAMP=.venv\.requirements-installed"

rem ---- 1. virtual environment
if exist "%PY%" goto :venv_ok
echo [준비] 처음 실행 - 가상환경(.venv)을 만듭니다.
where uv >nul 2>nul
if not errorlevel 1 (
  uv venv --python ">=3.11" .venv || goto :fail
  goto :venv_new
)
set "BASE="
py -3 -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>nul && set "BASE=py -3"
if not defined BASE python -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>nul && set "BASE=python"
if not defined BASE (
  echo [오류] Python 3.11 이상이 필요합니다.
  echo        https://www.python.org/downloads/ 에서 설치할 때 "Add python.exe to PATH" 를 체크하세요.
  goto :fail
)
%BASE% -m venv .venv || goto :fail
:venv_new
if exist "%STAMP%" del "%STAMP%"
:venv_ok

rem ---- 2. packages (again whenever requirements.txt is newer than the stamp)
set "NEED=1"
if exist "%STAMP%" (
  "%PY%" -c "import os,sys; sys.exit(os.path.getmtime('requirements.txt') > os.path.getmtime(r'%STAMP%'))" && set "NEED=0"
)
if "%NEED%"=="1" (
  echo [준비] 패키지를 설치합니다. 처음에는 몇 분 걸릴 수 있습니다.
  where uv >nul 2>nul
  if not errorlevel 1 (
    uv pip install --python "%PY%" -r requirements.txt || goto :fail
  ) else (
    "%PY%" -m pip install --upgrade pip >nul
    "%PY%" -m pip install -r requirements.txt || goto :fail
  )
  type nul > "%STAMP%"
)

rem ---- 3. mode
echo.
echo   StreamOMR
echo   1) 이 컴퓨터에서만 (노트북 웹캠)       http://localhost:8000
echo   2) 스마트폰도 같이 사용 (같은 Wi-Fi)   https://^<이 컴퓨터 IP^>:8443
echo.
set "MODE=1"
set /p "MODE=번호를 고르세요 [1]: "
if "%MODE%"=="2" (
  set "ARGS=--https"
  set "URL=https://localhost:8443/"
) else (
  set "ARGS="
  set "URL=http://localhost:8000/"
)

echo.
echo 서버를 끄려면 이 창에서 Ctrl+C 를 누르거나 창을 닫으세요.
echo 스마트폰 모드에서 Windows 방화벽 허용 창이 뜨면 "허용"을 누르세요.
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep 3; Start-Process '%URL%'"
"%PY%" scripts\serve.py --data data %ARGS%
echo.
pause
exit /b 0

:fail
echo.
echo [오류] 위 메시지를 확인하세요.
pause
exit /b 1
