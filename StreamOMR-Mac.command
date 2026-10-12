#!/bin/bash
# StreamOMR launcher for macOS - double-click in Finder.
# First run creates .venv and installs requirements.txt; later runs start the server directly.
cd "$(dirname "$0")" || exit 1

pause_exit() { echo; read -r -p "Enter 키를 누르면 창을 닫습니다... " _; exit "${1:-1}"; }

PY=.venv/bin/python
STAMP=.venv/.requirements-installed

# ---- 1. virtual environment
if [ ! -x "$PY" ]; then
  echo "[준비] 처음 실행 - 가상환경(.venv)을 만듭니다."
  if command -v uv >/dev/null 2>&1; then
    uv venv --python ">=3.11" .venv || pause_exit
  else
    BASE=""
    for c in python3.13 python3.12 python3.11 python3; do
      if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
        BASE="$c"; break
      fi
    done
    if [ -z "$BASE" ]; then
      echo "[오류] Python 3.11 이상이 필요합니다."
      echo "       https://www.python.org/downloads/ 에서 설치한 뒤 다시 실행하세요."
      pause_exit
    fi
    "$BASE" -m venv .venv || pause_exit
  fi
  rm -f "$STAMP"
fi

# ---- 2. packages (again whenever requirements.txt changes)
if [ ! -f "$STAMP" ] || [ requirements.txt -nt "$STAMP" ]; then
  echo "[준비] 패키지를 설치합니다. 처음에는 몇 분 걸릴 수 있습니다."
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY" -r requirements.txt || pause_exit
  else
    "$PY" -m pip install --upgrade pip >/dev/null
    "$PY" -m pip install -r requirements.txt || pause_exit
  fi
  touch "$STAMP"
fi

# ---- 3. mode
echo
echo "  StreamOMR"
echo "  1) 이 컴퓨터에서만 (노트북 웹캠)       http://localhost:8000"
echo "  2) 스마트폰도 같이 사용 (같은 Wi-Fi)   https://<이 컴퓨터 IP>:8443"
echo
read -r -p "번호를 고르세요 [1]: " MODE
if [ "$MODE" = "2" ]; then
  ARGS=(--https); URL="https://localhost:8443/"
else
  ARGS=(); URL="http://localhost:8000/"
fi

echo
echo "서버를 끄려면 이 창에서 Ctrl+C 를 누르거나 창을 닫으세요."
( sleep 3; open "$URL" ) &
"$PY" scripts/serve.py --data data "${ARGS[@]}"
pause_exit $?
