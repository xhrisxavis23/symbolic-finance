"""`run_e0.py` 의 `--teacher` CLI 배선 검증.

이 파일이 생기기 전에는 `run_e0.py` 를 부르는 테스트가 전혀 없었다(`grep -rn
"run_e0" tests/` 는 docstring 언급 한 줄뿐이었다) — `main()` 안 `if args.teacher
== "deeplob": ...` 분기가 깨져도(오타·조건 뒤집힘 등) 어떤 자동 테스트도 못
잡는다는 뜻이다. 실행(run_e0.py 자체)이 느려서 무거운 통합 테스트로 이 배선을
덮기는 어렵지만, `universe`/`e0_split`/`runner.run_law`/`e0_report.write` 를
가짜로 바꾸면 `main()` 의 인자 파싱→교사 선택 로직만 빠르게 격리해 볼 수 있다.

`--teacher deeplob` 인데 실제로는 `ScalarTeacher` 가 쓰이는 뮤테이션(조건이
항상 거짓이 되게 바꾸는 것 등)을 이 테스트로 직접 재현해 FAILED 를 확인했다
(자기검토, 2026-09-08) — 원복 후 md5 로 무변경을 재확인했다."""

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import run_e0  # noqa: E402
from sd.e0.teacher import ScalarDeepLOB, ScalarTeacher  # noqa: E402


class _FakeSplit:
    fit = ("A", "B")
    select = ("C",)
    per_stratum_counts = {"s1": {"fit": 2, "select": 1, "n": 3}}


class _FakeResult:
    tracks: dict = {}
    symbols_used = ("A", "B")
    select_symbols_used = ("C",)
    n_rows_total = 1
    recovered = False


def _patch_pipeline_scaffolding(monkeypatch, captured):
    """`main()` 이 실제 종목·데이터를 만지지 않게 앞뒤를 가짜로 감싼다.
    검증 대상(교사 선택 로직)만 진짜로 실행된다."""
    monkeypatch.setattr(run_e0.universe, "stock_symbols", lambda date: ["A", "B", "C"])
    monkeypatch.setattr(run_e0.universe, "liquidity_stats",
                        lambda symbols, date: pd.DataFrame(index=list(symbols)))
    monkeypatch.setattr(run_e0.universe, "assign_strata",
                        lambda stats: pd.DataFrame({"stratum": ["s1"] * len(stats)},
                                                   index=stats.index))
    monkeypatch.setattr(run_e0.universe, "slice_symbols",
                        lambda strata, per_stratum: list(strata.index))
    monkeypatch.setattr(run_e0.e0_split, "split_symbols",
                        lambda symbols, stratum_of, seed, fit_fraction: _FakeSplit())

    def fake_run_law(law, fit_symbols, select_symbols, date, *, seed, bottleneck, epochs,
                     sr_niterations, sr_maxsize, max_manifold_samples, teacher_cls,
                     teacher_window):
        captured["teacher_cls"] = teacher_cls
        captured["teacher_window"] = teacher_window
        return _FakeResult()

    monkeypatch.setattr(run_e0.runner, "run_law", fake_run_law)
    monkeypatch.setattr(run_e0.e0_report, "write",
                        lambda run_dir, results, symbols, args, symbol_split: Path("/dev/null"))


def test_teacher_deeplob_flag_actually_selects_scalardeeplob(monkeypatch):
    """이 테스트가 이 배선의 존재 이유다 — `--teacher deeplob` 을 줬는데 실제로
    `runner.run_law` 에 넘어가는 `teacher_cls` 가 `ScalarDeepLOB` 을 안 만들면
    (조건이 뒤집히거나 오타 나면) FAILED 여야 한다."""
    captured: dict = {}
    _patch_pipeline_scaffolding(monkeypatch, captured)
    monkeypatch.setattr(sys, "argv",
                        ["run_e0.py", "--laws", "L1", "--teacher", "deeplob",
                         "--teacher-window", "8"])

    rc = run_e0.main()

    assert rc == 0
    assert "teacher_cls" in captured, "run_law 이 한 번도 안 불렸다"
    instance = captured["teacher_cls"](n_features=3, bottleneck=2, seed=0)
    assert isinstance(instance, ScalarDeepLOB), (
        f"--teacher deeplob 인데 실제로 만들어진 교사가 {type(instance).__name__} 다")
    assert instance.window == 8
    assert captured["teacher_window"] == 8


def test_teacher_shallow_default_selects_scalarteacher_and_window_one(monkeypatch):
    """`--teacher` 를 생략하면(기본값 shallow) 여전히 `ScalarTeacher`, `teacher_window=1`
    이어야 한다 — 이 값이 `_maybe_window` 의 항등 분기(window<=1)를 태우는
    유일한 값이다(PREREG-E0-V2.md §1-2, sd/e0/runner.py `_maybe_window`)."""
    captured: dict = {}
    _patch_pipeline_scaffolding(monkeypatch, captured)
    monkeypatch.setattr(sys, "argv", ["run_e0.py", "--laws", "L1"])

    rc = run_e0.main()

    assert rc == 0
    assert captured["teacher_cls"] is ScalarTeacher
    assert captured["teacher_window"] == 1
