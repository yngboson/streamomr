# StreamOMR

실시간(스트리밍) 카메라 OMR. 서버판(FastAPI + 파이썬 OMR)과 브라우저 단독판(Pyodide)이 있다.

**🔗 라이브 데모 (GitHub Pages): https://yngboson.github.io/streamomr/**

## 에이전트 핸드오프 / 작업 로그
다음 에이전트는 여기부터 읽을 것. 새 작업은 아래 로그 맨 위에 추가.

### 구조
- `omr/` 인식 코어, `app/` 서버(`scripts/serve.py`), `standalone/` 브라우저 단독판 소스
- `scripts/build_standalone.py` → `standalone/streamomr.html` 생성 (+ `app/static/standalone.html` 복사)

### 아이폰에서 테스트하는 법
iOS Safari 카메라는 **https에서만** 동작 (`file://` 불가).
1. **GitHub Pages (권장):** https://yngboson.github.io/streamomr/ → `standalone/streamomr.html`로 리다이렉트. Pages 소스는 `main` 브랜치 루트(`.nojekyll` 있음). 플로우: `build_standalone.py` 실행 → 커밋/푸시 → 1~2분 후 반영.
2. **같은 Wi-Fi:** `python scripts/serve.py --data data --https` → 폰에서 `https://<맥 LAN IP>:8443` (자체 서명 인증서 경고 통과 필요). 서버측 파이썬 OMR을 그대로 테스트.
3. **터널:** ngrok / `cloudflared tunnel --url https://localhost:8443`.

### 알려진 한계
- standalone은 완전 오프라인이 아님: `standalone/worker.js`가 Pyodide 0.27.7 + numpy/OpenCV(약 25MB)를 jsdelivr CDN에서 받음 (`build_standalone.py`의 `PYODIDE_BASE`). 진짜 오프라인화하려면 Pyodide를 레포에 번들해야 함.
- iPhone Safari에서 Worker 안 Pyodide 메모리 한계로 탭이 죽을 가능성 있음 — 실기기 검증 전.

### 로그
- 2026-10-12: 실사진 기준 세트 (realasset/*.jpeg 8장, 아이폰 24MP, 4등분 접힌 v2 시트 — 사진은 레포 미포함, 정답은 `realasset/truth.json` 형식, `scripts/photo_bench.py`/`tests/test_real_photos.py`). 3개 에이전트 병렬 후 통합: (1) `detect.find_markers_scaled` — 긴 변 1920px 초과 프레임은 축소본에서 마커 검출 후 원본에서 코너 보정 (24MP 548→~210ms), `_box_mean` clip 제거; (2) `detect._complete` — 마커 3개만 찾으면 4번째 예상 위치를 좁은 창에서 재탐색(45° 회전/번진 마커); 자기 페이지 마커와 겹치는 다른 페이지 마커 무시(`mixed_page_markers` 오판 제거); (3) `_fit_alignment` — 트랙으로 평탄화되고 링 인라이어 ≥0.8이면 최대 이동 허용 2→4mm. 결과: 5/8 → 8/8 (scale 1/0.5/0.33, 회전 90/180/270), 최대 ~250ms. 전체 테스트 302 통과, 학번 오독 0. geom/illum 벤치: 판독률↑, 조용한 오류 +1씩(새로 구제된 프레임의 흐린 마크 문항 — 흐린 마크 규칙 쪽 과제). 24MP JPEG 디코드(280–550ms)는 recognize 시간에 미포함.
- 2026-10-12: 접힌/말린 종이 대응 — 시트 4변에 타이밍 트랙(검은 눈금, 6mm 피치, 좌우 40개·상하 26개) 추가, 헤더 version 2로 구분(`Template.tracks` 기본 True, `tracks=False`는 기존 v1 그대로, 옛 레이아웃/인쇄런은 트랙 없음으로 로드). `omr/tracks.py`가 코너마커 워프 위에서 눈금을 DP로 추적해 네 변 변위를 Coons 패치로 보간하고 캔버스를 평탄화한 뒤 기존 링 정렬/판정 사용. 링 확인 실패 시 `alignment_unverified`, 행 단위 링 점검으로 국소 휨은 애매 처리, 트랙 미검출 시 기존 2차 모델 폴백. `scripts/geom_bench.py`(fold/fold_curl/fold_hand/fold_big) 전→후: 판독 프레임 fold 72→90%, fold_curl 62→78%, curl 60→85%, raking 68→92%, fold_big 정확도 71→96%, 학번 오독 0 유지, 실영상 회귀 없음. 한계: 가장자리 25mm 이내 접힘은 검토 표시만, 상하 눈금은 가장자리 4.4mm(프린터 클리핑 가능), 서버 UI에 tracks 옵션 없음.
- 2026-10-12: 약한 연필 마킹(반사 아래 ~20% 어두움) 상대 규칙 — 같은 문항의 다른 선지 중앙값보다 원 전체 어두움(`_mean_dark`, 원이 고르게 어두울 때만)이 0.15 이상 크면 마킹(`omr/recognize.py` `_relative_marks`/`_decide`). 절대 규칙이 확신 못 한 행에만 추가, 0.15~0.20은 플래그, 2개 이상이면 multi. 지우개 흔적과 구분이 안 돼 `_rel_page_trust`(문항 40% 이상이 상대 마크, ghost 거의 없음)일 때만 켬. `scripts/faint_bench.py`: faint 정독 64.8→99.8%, 조용한 오류 219→0, 학번 0→71%(오독 0). **알려진 문제: 반사가 시트 일부에만 있으면 40% 조건을 못 넘어 규칙이 꺼짐 — 영역 단위 게이트 필요.**
- 2026-10-12: 조명에 약한 헤더 인식 개선 (`scripts/header_bench.py`로 조건별 헤더 디코드율 측정, 합성 40프레임/조건, 전→후). 진단: 실패 대부분은 밝기 문제가 아니라 마커 3개만 보여 외삽된 호모그래피 때문에 헤더 한쪽 끝이 2~3mm 어긋나 `header_reg_mm` 검사에서 탈락한 것. 변경: (1) `recognize._flatten_header` 열 단위 종이 밝기 평탄화(효과 작음, 유지), (2) 1차 실패 시 탐색창/허용오차를 넓힌 재시도 + 행을 따라가는 `_track_dy`(CRC-16이 판정, 성공 시 `header_wide`/`header_tracked` quality 플래그), (3) `detect.flatten_lighting` 마커 검출 최후 수단 패스(마커 후보가 보일 때만 실행해서 빈 화면 비용 불변). 결과: shadow 78→100%, glare 88→100%, h_multi 68→88%, h_dimgrad 42→90%, h_all 28→52%, curl 52→62%, raking 55→62%. clean/dark 100% 유지, 전체 테스트 249 통과. 남은 실패: curl/raking/h_all은 조명이 아니라 종이 휨 + 마커 3개 외삽(헤더가 5mm 이상 기울어짐) — 비선형 보정이 필요. 서버판/standalone 모두 같은 `omr/`를 쓰므로 standalone은 재빌드함.
- 2026-10-12: standalone "최근 스캔" 항목을 탭하면 상세 모달(`#detail`)이 열림. `standalone/glue.py::detail()`이 커밋 시 보정된 시트(JPEG base64)와 문항/학번/과목별 버블 좌표·점수·판정을 내려주고, `standalone/app.js`의 `showDetail/renderDetail/drawSheet`가 시트 전체를 한 캔버스에 그리고(크롭 없음) 모든 버블에 색 링 + 문제 행에 박스, 아래에 문제 항목 목록(초록=인식됨, 빨강=복수마킹, 주황=애매/공란)과 선지별 농도(%)를 표시. **브라우저 실기기 렌더링은 미검증**(파이썬 쪽 detail()과 JS 문법만 확인). 서버판 `app/static/scan.js`에는 아직 없음.
- 2026-10-12: GitHub Pages 배포 경로 정리(루트 `index.html` 리다이렉트 추가), 이 핸드오프 섹션 작성. 아이폰 실기기 테스트는 아직 안 함.
