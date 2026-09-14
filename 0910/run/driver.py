#!/usr/bin/env python3
"""E1 실행 드라이버. 이것 하나를 `setsid nohup` 으로 띄운다.

    setsid nohup /opt/conda/envs/lab/bin/python3 -m run.driver \
        > /home/dgu/tick/symbolic/0910/run/driver.log 2>&1 < /dev/null &
    echo $! > /home/dgu/tick/symbolic/0910/run/driver.pid

각 스테이지는 자기 완료 파일이 있으면 스킵한다(재개 가능) — 드라이버가
죽어도 `python -m run.driver` 를 다시 돌리면 끝난 것은 건너뛰고 이어간다.

스테이지:
  1. seed 0·1·2 교사+SR+컴파일 (병렬 3프로세스)
  2. M 선택 (전 시드 통합 재생)
  3. L/M/H 저하 곡선 + 최종 판정
  4. RESULTS.md 초안 작성 + 커밋
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from . import locked_params as P
from .common import STATE_DIR, log

RUN_DIR = Path("/home/dgu/tick/symbolic/0910/run")
PY = "/opt/conda/envs/lab/bin/python3"


def _run_module(module: str, args: list[str], log_name: str) -> int:
    log_path = RUN_DIR / f"{log_name}.log"
    with open(log_path, "a") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} 시작: {module} {args} ===\n")
        fh.flush()
        proc = subprocess.run([PY, "-m", module, *args], cwd="/home/dgu/tick/symbolic/0910",
                              stdout=fh, stderr=subprocess.STDOUT)
    return proc.returncode


def stage_seeds() -> bool:
    """`True` 를 돌려주면 다음 스테이지로 간다 — **3개 전부 성공했다는 뜻이
    아니라 최소 1개는 있다는 뜻이다.** R37-1(3개 전부 통과해야 성공 주장)은
    최종 판정(stage_curve)에서 걸리지, 여기서 미리 막지 않는다 — 한 시드가
    복구 불가능하게 실패해도 나머지로 부분 결과를 봐야 무엇이 잘못됐는지
    알 수 있다(세션 중단으로 작업을 잃은 전례가 있다 — 부분 결과라도
    남기는 쪽이 낫다)."""
    procs: dict[int, subprocess.Popen] = {}
    for seed in P.SEEDS:
        if (STATE_DIR / f"seed_{seed}.json").exists():
            log("driver", f"seed={seed} 이미 완료 — 스킵")
            continue
        log_path = RUN_DIR / f"seed_{seed}.log"
        fh = open(log_path, "a")
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} 시작 ===\n")
        fh.flush()
        proc = subprocess.Popen([PY, "-m", "run.stage_seed", "--seed", str(seed)],
                                cwd="/home/dgu/tick/symbolic/0910",
                                stdout=fh, stderr=subprocess.STDOUT)
        procs[seed] = proc
        log("driver", f"seed={seed} 프로세스 시작 pid={proc.pid}")

    for seed, proc in procs.items():
        rc = proc.wait()
        log("driver", f"seed={seed} 종료 rc={rc}"
                      + ("" if rc == 0 else " — 실패, seed_{}.log 확인".format(seed)))

    n_ok = sum(1 for s in P.SEEDS if (STATE_DIR / f"seed_{s}.json").exists())
    log("driver", f"시드 완료 {n_ok}/{len(P.SEEDS)}")
    if n_ok < len(P.SEEDS):
        log("driver", "경고: 일부 시드가 실패했다 — R37-1 '3개 전부 통과' 성공 "
                      "주장은 이 실행에서 불가능하다. 부분 결과로 계속 진행한다")
    return n_ok >= 1


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log("driver", "E1 실행 시작")

    if not stage_seeds():
        log("driver", "치명적 오류: 3개 시드 전부 실패했다(state/seed_*.json 이 "
                      "하나도 없다). seed_N.log 를 확인하라. 다시 실행하면 처음부터 "
                      "재시도한다.")
        return 1
    log("driver", f"전 시드 완료 ({time.time()-t0:.0f}s 경과)")

    rc = _run_module("run.stage_select", [], "select")
    if rc != 0:
        log("driver", f"치명적 오류: stage_select 실패 rc={rc}. select.log 참조.")
        return 1
    log("driver", f"M 선택 완료 ({time.time()-t0:.0f}s 경과)")

    rc = _run_module("run.stage_curve", [], "curve")
    if rc != 0:
        log("driver", f"치명적 오류: stage_curve 실패 rc={rc}. curve.log 참조.")
        return 1
    log("driver", f"L/M/H 저하 곡선 완료 ({time.time()-t0:.0f}s 경과)")

    _run_module("run.write_results", [], "write_results")

    total = time.time() - t0
    log("driver", f"E1 실행 전체 완료. 총 {total:.0f}초 ({total/3600:.2f}시간)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
