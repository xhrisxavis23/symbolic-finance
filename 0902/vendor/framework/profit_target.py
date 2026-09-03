"""모든 실행 후보가 공유하는 비용 포함 수익 목표."""

from __future__ import annotations

from typing import Any, Mapping

from . import canonical as K
from .config import sha256_json


SCHEMA_VERSION = "profit_target.v1"
SEARCH_PRIMARY_METRIC = "CANONICAL_FIXED_EXIT_TOTAL_NET_BPS"


def canonical_target() -> dict[str, Any]:
    """V9 실제 체결가 회계와 총 순이익 판정을 한 계약으로 고정한다."""
    cost = float(K.CANONICAL["cost"]["explicit_cost_bps"])
    return {
        "schema": SCHEMA_VERSION,
        "population": "FILLED",
        "outcome": "gross_bps_from_actual_fill_to_exit",
        "fill_price_basis": "ACTUAL_BID1_ASK1",
        "spread_mode": "IMPLICIT_IN_FILL_PRICE",
        "explicit_cost_bps_per_fill": cost,
        "hurdle_bps": cost,
        "primary_metric": "net_bps_total",
        "selection_objective": "MAXIMIZE",
        "eligibility_condition": {
            "metric": "mean_gross_bps_per_fill", "operator": ">", "value": cost,
        },
        "success_condition": {
            "metric": "net_bps_total", "operator": ">", "value": 0.0,
        },
    }


def target_hash(target: Mapping[str, Any] | None = None) -> str:
    return sha256_json(dict(target or canonical_target()))[:16]


def validate(target: Any) -> dict[str, Any]:
    expected = canonical_target()
    problems: list[str] = []
    if not isinstance(target, Mapping):
        problems.append("profit_target이 객체가 아니다")
    elif dict(target) != expected:
        problems.append("profit_target이 정본 V9 비용 포함 총 순이익 계약과 다르다")
    return {"ok": not problems, "problems": problems,
            "expected_sha256": target_hash(expected)}


def require(container: Mapping[str, Any], where: str) -> dict[str, Any]:
    target = container.get("profit_target")
    report = validate(target)
    if not report["ok"]:
        raise ValueError(f"{where}: " + "; ".join(report["problems"]))
    return dict(target)


def evaluate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """백테스트 합계가 계약을 넘었는지 같은 분모로 판정한다."""
    target = canonical_target()
    fills = int(metrics.get("fills") or 0)
    net = float(metrics.get("net_bps_total") or 0.0)
    gross = float(metrics.get("gross_bps_total") or 0.0)
    cost = float(target["explicit_cost_bps_per_fill"])
    expected_net = gross - cost * fills
    identity_ok = abs(net - expected_net) <= 1e-6
    mean_gross = gross / fills if fills else None
    hurdle_passed = mean_gross is not None and mean_gross > cost
    return {
        "schema": "profit_target_evaluation.v1",
        "profit_target_sha256": target_hash(target),
        "population_count": fills,
        "net_bps_total": net,
        "mean_gross_bps_per_fill": mean_gross,
        "hurdle_bps": cost,
        "eligibility_condition_passed": hurdle_passed,
        "primary_condition_passed": net > 0.0,
        "accounting_identity_ok": identity_ok,
        "success": fills > 0 and identity_ok and hurdle_passed and net > 0.0,
    }
