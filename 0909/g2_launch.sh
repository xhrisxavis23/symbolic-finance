#!/bin/bash
# G2 전체 실행 런처. Ruling R32(코디네이터 승인) 조건 이행:
#   - 다른 작업자의 E0 두 개가 끝날 때까지 블로킹 대기(짧은 폴링 금지,
#     코디네이터가 지정한 정확히 이 루프)
#   - 대기·본 실행 둘 다 nohup+setsid 로 세션과 분리 — 세션이 끊겨도 산다
#   - PID 를 파일로 남긴다
set -uo pipefail
cd /home/dgu/tick/symbolic/0909

echo "[$(date '+%H:%M:%S')] 대기 시작 — run_e0.py --teacher 프로세스가 끝날 때까지" >> g2_launch.log
while pgrep -f "run_e0.py --teacher" >/dev/null; do sleep 120; done
echo "[$(date '+%H:%M:%S')] 대기 종료 — 다른 작업자의 E0 실행이 끝났다. G2 전체 실행을 시작한다." >> g2_launch.log

nohup python3 g2_full_run.py >> g2_full_run.log 2>&1 &
echo $! > g2_full_run.pid
echo "[$(date '+%H:%M:%S')] G2 전체 실행 시작. PID=$(cat g2_full_run.pid)" >> g2_launch.log
