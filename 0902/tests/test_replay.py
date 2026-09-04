import sys
from pathlib import Path

import pandas as pd
import pytest
import sympy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import config, replay  # noqa: E402
from sd.compile import compile_candidates  # noqa: E402
from sd.sr.base import Candidate  # noqa: E402


def _entries():
    expr = sympy.Symbol("book_imbalance")
    candidate = Candidate(expr=expr, complexity=1, in_sample_score=0.5,
                          backend="naive", seed=0)
    ok, _failed = compile_candidates([candidate])
    return ok


def _two_entries():
    """서로 다른 정규형으로 붕괴되지 않는 두 후보 — attempt_count 가 entries 수에도
    실제로 반응하는지 보려면 entries=1 인 픽스처만으로는 부족하다 (len(grid) 만
    반환해도 entries=1 일 때는 우연히 같은 값이 나온다).

    `spread_bps` 대신 `spread_to_round_trip_cost_ratio` 를 쓴다 — 1단계에서
    `check.static_check` 에 `catalog.condition_problem` 검사가 들어가면서
    `spread_bps` 는 (`THRESHOLD_DERIVED_INPUT_ONLY` — 절대 원값 조건 금지) 더
    이상 임계 조건의 직접 대상이 될 수 없다. 이 테스트는 `attempt_count` 의
    산술만 보는 것이 목적이라, 여전히 직접 조건으로 허용되고 `book_imbalance`
    와 다른 정규형으로 남는 feature 로 바꾼다."""
    bi = sympy.Symbol("book_imbalance")
    ratio = sympy.Symbol("spread_to_round_trip_cost_ratio")
    candidates = [
        Candidate(expr=bi, complexity=1, in_sample_score=0.5, backend="naive", seed=0),
        Candidate(expr=ratio, complexity=1, in_sample_score=0.5, backend="naive", seed=0),
    ]
    ok, _failed = compile_candidates(candidates)
    return ok


def test_attempts_counts_entries_times_grid(tmp_path):
    entries = _entries()
    assert replay.attempt_count(entries, config.QUANTILE_GRID) == len(config.QUANTILE_GRID)


def test_attempts_scales_with_entry_count_not_just_grid():
    """`attempt_count` 가 상수나 `len(grid)` 만 반환해도 entries=1 짜리 테스트는
    우연히 통과한다. entries 를 2개로 늘려 실제로 곱셈이 일어나는지 본다."""
    entries = _two_entries()
    assert len(entries) == 2
    grid = (0.70, 0.85, 0.95)
    assert replay.attempt_count(entries, grid) == 6
    assert replay.attempt_count(entries, grid) != replay.attempt_count(_entries(), grid)


def test_run_rejects_empty_entries(tmp_path):
    with pytest.raises(ValueError):
        replay.run([], symbols=("005930",), date=config.DATE,
                   output_dir=tmp_path, grid=(0.85,), workers=1)


def test_run_raises_when_manifest_reports_errors(tmp_path, monkeypatch):
    """`errors` 가 비어 있지 않으면 원장 일부만으로 조용히 집계해서는 안 된다 —
    가드를 지우면 이 테스트가 FAILED 로 잡아야 한다."""
    def _fake_run_backtest(**kwargs):
        return {"errors": ["005930: 재생 실패"],
                "canonical_backtest_profile_hash": config.PROFILE_HASH}
    monkeypatch.setattr(replay._canonical, "run_backtest", _fake_run_backtest)

    with pytest.raises(RuntimeError):
        replay.run(_entries(), symbols=("005930",), date=config.DATE,
                   output_dir=tmp_path, grid=(0.85,), workers=1)


def test_run_passes_through_when_manifest_has_no_errors(tmp_path, monkeypatch):
    """위 테스트의 대응 짝 — 가드가 지나치게 넓어서 정상 매니페스트까지 막지
    않는지 확인한다."""
    def _fake_run_backtest(**kwargs):
        return {"errors": [], "canonical_backtest_profile_hash": config.PROFILE_HASH}
    monkeypatch.setattr(replay._canonical, "run_backtest", _fake_run_backtest)

    result = replay.run(_entries(), symbols=("005930",), date=config.DATE,
                        output_dir=tmp_path, grid=(0.85,), workers=1)
    assert result.manifest["errors"] == []


# -- F6 (최종 전체 리뷰): `contract_id` 는 분위를 소수 2자리로 포맷한다. 정본
# 격자 `(0.70, 0.85, 0.95)` 에서는 충돌하지 않지만, 임의 격자에서 두 분위가
# 같은 문자열로 포맷되면 `contracts` dict 항목이 조용히 덮어써져 계약 하나가
# 사라지는데 `attempt_count` 는 `len(entries)*len(grid)` 를 그대로 돌려준다 —
# 보고되는 N 이 실제보다 커진다. `replay.run` 이 계약 ID 를 만든 직후 유일성을
# 단언해야 한다.

def test_run_rejects_colliding_quantile_grid(tmp_path, monkeypatch):
    """`0.700001` 과 `0.704999` 는 소수 2자리로는 둘 다 `q0.70` 이 된다 —
    이런 격자로 재생을 부르면 조용히 계약이 사라지는 대신 명확한 예외가
    나야 한다. `run_backtest` 를 가짜로 바꿔둔다 — 충돌 검사가 없으면 이
    호출까지 실제로 도달해(정본 재생을 시작해) 뮤테이션 확인이 느려지는
    것을 막기 위함이다(검사가 살아있는 정상 경로에서는 애초에 도달하지
    않는다)."""
    def _fake_run_backtest(**kwargs):
        return {"errors": [], "canonical_backtest_profile_hash": config.PROFILE_HASH}
    monkeypatch.setattr(replay._canonical, "run_backtest", _fake_run_backtest)

    colliding_grid = (0.700001, 0.704999, 0.95)
    with pytest.raises(ValueError, match="계약 ID 충돌"):
        replay.run(_entries(), symbols=("005930",), date=config.DATE,
                  output_dir=tmp_path, grid=colliding_grid, workers=1)


def test_run_accepts_the_canonical_grid_without_collision(tmp_path, monkeypatch):
    """정본 격자 `(0.70, 0.85, 0.95)` 는 충돌하지 않는다 — 위 테스트가
    과하게 넓어서 정상 격자까지 막지 않는지 확인한다."""
    def _fake_run_backtest(**kwargs):
        return {"errors": [], "canonical_backtest_profile_hash": config.PROFILE_HASH}
    monkeypatch.setattr(replay._canonical, "run_backtest", _fake_run_backtest)

    result = replay.run(_entries(), symbols=("005930",), date=config.DATE,
                        output_dir=tmp_path, grid=(0.70, 0.85, 0.95), workers=1)
    assert len(result.contract_ids) == 3


def test_contract_ids_differ_across_quantile_grid(tmp_path, monkeypatch):
    """분위마다 다른 계약 ID 를 만드는가 — 전부 같은 ID 를 쓰면 격자가 무의미해진다."""
    captured = {}

    def _fake_run_backtest(**kwargs):
        captured.update(kwargs)
        return {"errors": [], "canonical_backtest_profile_hash": config.PROFILE_HASH}
    monkeypatch.setattr(replay._canonical, "run_backtest", _fake_run_backtest)

    grid = (0.70, 0.85, 0.95)
    entries = _entries()
    replay.run(entries, symbols=("005930", "000660"), date=config.DATE,
              output_dir=tmp_path, grid=grid, workers=1)

    contracts = captured["contracts"]
    assert len(contracts) == len(entries) * len(grid)          # 계약마다 별개 항목
    assert len(set(contracts)) == len(entries) * len(grid)     # ID 가 실제로 서로 다름


def test_parameter_table_covers_every_contract_symbol_pair(tmp_path, monkeypatch):
    """`parameter_table` 이 모든 (계약, 종목) 조합을 덮는가."""
    captured = {}

    def _fake_run_backtest(**kwargs):
        captured.update(kwargs)
        return {"errors": [], "canonical_backtest_profile_hash": config.PROFILE_HASH}
    monkeypatch.setattr(replay._canonical, "run_backtest", _fake_run_backtest)

    grid = (0.70, 0.85)
    symbols = ("005930", "000660", "035420")
    entries = _entries()
    result = replay.run(entries, symbols=symbols, date=config.DATE,
                        output_dir=tmp_path, grid=grid, workers=1)

    table = captured["parameter_table"]
    expected_keys = {f"{cid}:{sym}:{config.DATE}"
                     for cid in result.contract_ids for sym in symbols}
    assert set(table) == expected_keys
    assert len(table) == len(entries) * len(grid) * len(symbols)


@pytest.mark.slow
def test_replay_produces_a_ledger_with_no_errors(tmp_path):
    result = replay.run(_entries(), symbols=("000660", "035420"), date=config.DATE,
                        output_dir=tmp_path, grid=(0.85,), workers=2)
    assert result.manifest["errors"] == []
    assert result.manifest["canonical_backtest_profile_hash"] == config.PROFILE_HASH
    frame = pd.read_parquet(result.ledger_path)
    assert len(frame) > 0
    assert set(frame["status"].unique()) <= {"FILLED", "UNFILLED", "CENSORED"}
    assert result.attempts == 1
