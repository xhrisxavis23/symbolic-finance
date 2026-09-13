#!/bin/bash
# 종목군 실행 전용 감시: 진행 JSON 의 단계가 바뀔 때마다 커밋한다(대용량 산출물은 .gitignore 로 제외).
# scale/run_detached.sh 는 돌고 있는 두 감시가 읽는 중이라 고치지 않고 따로 만든다.
set -u
NAME="$1"; OUT="$2"; shift 2
cd /home/dgu/tick/symbolic/0911
LOG="groups/${NAME}.log"
echo "=== [$(date -u +%F_%T)] 기동: $* (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-} PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-}) ===" >> "$LOG"
python3 -u "$@" >> "$LOG" 2>&1 &
PY=$!
echo "$PY" > "groups/${NAME}.pid"
last=""
commit() {
  [ -f "$OUT" ] || return 0
  cur=$(md5sum "$OUT" | cut -d' ' -f1)
  [ "$cur" = "$last" ] && return 0
  stage=$(python3 -c "import json;print(json.load(open('$OUT')).get('stage','?'))")
  (
    flock 9
    git add "$OUT" "$LOG" >/dev/null 2>&1
    git commit -q -m "data(0911): ${NAME} 단계 — ${stage}

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JVLoKpNP3EW1ya1hXnfX9e" >/dev/null 2>&1
  ) 9>"/tmp/claude-1003/-home-dgu-tick-symbolic/4a3e1c4b-3ad5-47d8-a5f5-5cdf1d5a7bb6/scratchpad/0911-git.lock"
  last="$cur"
}
while kill -0 "$PY" 2>/dev/null; do commit; sleep 60; done
commit
wait "$PY"; rc=$?; echo "=== [$(date -u +%F_%T)] 종료 exit=$rc ===" >> "$LOG"
( flock 9; git add "$LOG" >/dev/null 2>&1; git commit -q -m "data(0911): ${NAME} 종료 exit=${rc}

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JVLoKpNP3EW1ya1hXnfX9e" >/dev/null 2>&1 ) 9>"/tmp/claude-1003/-home-dgu-tick-symbolic/4a3e1c4b-3ad5-47d8-a5f5-5cdf1d5a7bb6/scratchpad/0911-git.lock"
