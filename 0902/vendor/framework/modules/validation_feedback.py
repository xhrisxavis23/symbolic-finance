"""Validation 탈락 거래 장부를 다음 연구의 개발 피드백으로 제한해 연다.

이 모듈은 이미 끝난 Validation Backtest의 음수 총손익만 읽는다. 다음 Agent가
부모 알고리즘의 일반화 실패를 고칠 질문을 정하게 할 뿐, Feature Profile Evidence나
새 가설의 지지 증거가 되지는 않는다. Final Backtest는 절대로 읽지 않는다.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..config import clean
from ..contracts.artifacts import create_artifact, read_artifact, write_artifact
from . import discovery_loss


SCHEMA_VERSION = "development_validation_feedback.v1"
ARTIFACT_NAME = "development_validation_feedback_artifact.json"


def build(source_research: Path, hypothesis_id: str, output: Path) -> dict[str, Any] | None:
    """총손익 음수로 끝난 Validation ledger만 다음 세대에 요약한다.

    Validation을 통과했거나 아직 실행하지 않은 부모는 고칠 실패가 없으므로 ``None``을
    돌려준다. Final artifact는 이 함수의 입력이나 탐색 대상이 아니다.
    """
    source_research = Path(source_research).resolve()
    run = read_artifact(source_research / "research_run.json", kind="research_run")
    refs = dict(run["payload"].get("artifacts") or {})
    validation_ref = str(refs.get("validation_backtest_set") or "")
    if not validation_ref:
        return None
    validation_path = source_research / "06_validation_backtest" / "validation_artifact.json"
    if not validation_path.is_file():
        return None
    validation = read_artifact(validation_path, kind="validation_backtest_set")
    if validation["artifact_id"] != validation_ref:
        raise ValueError("research run과 Validation Backtest artifact의 lineage가 다르다")
    unit = (validation["payload"].get("units") or {}).get(str(hypothesis_id))
    if not isinstance(unit, Mapping):
        return None
    backtest = unit.get("backtest") or {}
    metrics = backtest.get("metrics") or {}
    total = _number(metrics.get("net_bps_total"))
    if (backtest.get("state") != "BACKTEST_COMPLETE"
            or total is None or total > 0.0):
        return None
    ledger_path = source_research / "06_validation_backtest" / "units" / str(hypothesis_id) / "ledger.parquet"
    if not ledger_path.is_file():
        raise FileNotFoundError(f"Validation feedback ledger가 없다: {ledger_path}")
    frame = pd.read_parquet(ledger_path)
    if "diagnostic_cohort" not in frame.columns:
        raise ValueError("Validation ledger에 diagnostic_cohort가 없다")
    profile = clean(((run["payload"].get("request") or {}).get("profile") or {}))
    payload = {
        "schema": SCHEMA_VERSION,
        "source_hypothesis_id": str(hypothesis_id),
        "profile": profile,
        "development_feedback_only": True,
        "source": {
            "research_run_artifact": run["artifact_id"],
            "validation_backtest_artifact": validation["artifact_id"],
            "ledger": str(ledger_path),
            "validation_dates": list(backtest.get("dates") or validation["payload"].get("dates") or []),
            "parameter_lock": dict(backtest.get("parameter_lock") or {}),
            "entry_guards": [dict(item) for item in backtest.get("entry_guards") or []],
            "net_bps_total": total,
        },
        "interpretation_boundary": (
            "이 context는 앞선 후보가 Validation Backtest에서 총손익 0 이하였다는 개발 피드백이다. "
            "사례와 수치는 다음 entry 연구 질문을 고르는 데만 쓰며, 새 가설의 Feature Profile Evidence, "
            "검증 증거, 예측 또는 수익 근거가 아니다. 이 날짜는 후속 세대에서 재사용되는 개발용 날짜로 "
            "취급하며, Final Backtest는 이 context에 포함되지 않고 읽히지도 않는다."
        ),
        "summary": discovery_loss._summary(frame),
        "case_samples": discovery_loss._cases(frame),
    }
    artifact = create_artifact("development_validation_feedback", payload, parents={
        "research_run": run["artifact_id"],
        "validation_backtest_set": validation["artifact_id"],
    })
    write_artifact(Path(output) / ARTIFACT_NAME, artifact)
    return artifact


def context_identity(source_research: Path, hypothesis_id: str) -> dict[str, Any]:
    """재시도 이력이 어느 Validation 실패를 이미 반영했는지 구분한다."""
    source_research = Path(source_research).resolve()
    run = read_artifact(source_research / "research_run.json", kind="research_run")
    refs = dict(run["payload"].get("artifacts") or {})
    validation_ref = str(refs.get("validation_backtest_set") or "")
    if not validation_ref:
        return {"schema": SCHEMA_VERSION, "state": "NOT_AVAILABLE"}
    path = source_research / "06_validation_backtest" / "validation_artifact.json"
    if not path.is_file():
        return {"schema": SCHEMA_VERSION, "state": "NOT_AVAILABLE"}
    validation = read_artifact(path, kind="validation_backtest_set")
    if validation["artifact_id"] != validation_ref:
        raise ValueError("research run과 Validation Backtest artifact의 lineage가 다르다")
    unit = (validation["payload"].get("units") or {}).get(str(hypothesis_id))
    backtest = unit.get("backtest") if isinstance(unit, Mapping) else None
    total = _number((backtest or {}).get("metrics", {}).get("net_bps_total"))
    if (not isinstance(backtest, Mapping) or backtest.get("state") != "BACKTEST_COMPLETE"
            or total is None or total > 0.0):
        return {"schema": SCHEMA_VERSION, "state": "NOT_FAILED"}
    return {
        "schema": SCHEMA_VERSION,
        "state": "NEGATIVE_TOTAL",
        "research_run_artifact": run["artifact_id"],
        "validation_backtest_artifact": validation["artifact_id"],
        "hypothesis_id": str(hypothesis_id),
        "net_bps_total": total,
        "entry_guards": [dict(item) for item in backtest.get("entry_guards") or []],
        "parameter_lock": dict(backtest.get("parameter_lock") or {}),
    }


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) and math.isfinite(number) else None
