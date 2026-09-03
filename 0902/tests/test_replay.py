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
    반환해도 entries=1 일 때는 우연히 같은 값이 나온다)."""
    bi = sympy.Symbol("book_imbalance")
    spread = sympy.Symbol("spread_bps")
    candidates = [
        Candidate(expr=bi, complexity=1, in_sample_score=0.5, backend="naive", seed=0),
        Candidate(expr=spread, complexity=1, in_sample_score=0.5, backend="naive", seed=0),
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
