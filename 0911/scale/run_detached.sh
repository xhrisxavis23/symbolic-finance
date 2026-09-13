#!/bin/bash
# 크기별 체크포인트가 갱신될 때마다 커밋한다. 재부팅이 와도 끝난 지점은 저장소에 남는다.
set -u
NAME="$1"; OUT="$2"; shift 2
cd /home/dgu/tick/symbolic/0911
LOG="scale/${NAME}.log"
echo "=== [$(date -u +%F_%T)] 기동: $* ===" >> "$LOG"
python3 -u "$@" >> "$LOG" 2>&1 &
PY=$!
echo "$PY" > "scale/pids/${NAME}.pid"
last=""
commit() {
  [ -f "$OUT" ] || return 0
  cur=$(md5sum "$OUT" | cut -d' ' -f1)
  [ "$cur" = "$last" ] && return 0
  n=$(python3 -c "import json;print(len(json.load(open('$OUT'))))")
  (
    flock 9
    git add "$OUT" "$LOG" >/dev/null 2>&1
    git commit -q -m "data(0911): ${NAME} 체크포인트 — ${n}개 지점 저장

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JVLoKpNP3EW1ya1hXnfX9e" >/dev/null 2>&1
  ) 9>"/tmp/claude-1003/-home-dgu-tick-symbolic/4a3e1c4b-3ad5-47d8-a5f5-5cdf1d5a7bb6/scratchpad/0911-git.lock"
  last="$cur"
}
while kill -0 "$PY" 2>/dev/null; do commit; sleep 60; done
commit
wait "$PY"; echo "=== [$(date -u +%F_%T)] 종료 exit=$? ===" >> "$LOG"
