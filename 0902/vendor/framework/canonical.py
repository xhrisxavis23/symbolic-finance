"""측정 장비 하나. 어떤 가설이 와도 이 아래는 똑같다.

    가설과 파라미터는 **신호**를 만든다.
    백테스트는 그 신호를 **언제나 같은** 호가·큐 규칙으로 실행한다.

가설 선택에 쓰는 실행 방식은 하나다 — `CANONICAL_QUEUE_V9`. 비교용 청산은 같은
진입 계약을 별도 원장으로 재생할 뿐, 선택·승격 기준을 바꾸지 않는다.

## 스프레드를 어떻게 다루는가

    체결가격  실제 BID/ASK 에서 난다  →  스프레드가 여기 이미 들어 있다
    명시비용  수수료·세금             →  따로 뺀다

그래서 `net = (청산체결 / 진입체결 - 1) * 1e4 - 수수료` 다. **`spread_bps` 를 또 빼지
않는다.** 빼면 같은 비용을 두 번 무는 것이다.

## 큐를 어떻게 다루는가

    호가가 다음 레벨로 갔다는 이유만으로 체결시키지 않는다.
    내 앞의 물량이 실제로 소진되어야 내 차례가 온다.

정확한 거래소 모사가 아니다. 스냅샷 데이터로 알 수 있는 것만 쓰는 **보수적이고 일관된
근사**다. 주문 추가·취소·큐 순번은 이 데이터에 없다.

## 무엇이 탐색 대상이 아닌가

큐 모델·체결 가정·비용·결정 간격·주문 유지시간은 전부 **프로토콜**이다. 파라미터 탐색은
가설 명세가 연 신호 파라미터만 건드린다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import config as cfg, contract as contract_runtime, fill, ledger, metrics
from .config import now_utc, sha256_json, write_json


SCHEMA_VERSION = "canonical_backtest_profile.v9"
PROFILE_ID = "CANONICAL_QUEUE_V9"
V8_900_COMPARISON_PROFILE_ID = "CANONICAL_QUEUE_V8_900_COMPARISON"

# 관문 (§54).
READY_FOR_CANONICAL_BACKTEST_SEARCH = "READY_FOR_CANONICAL_BACKTEST_SEARCH"
# v1 -> v2: 손절(gross)을 넣었다. 프로필 해시가 달라지므로 v1 결과와 직접 비교하지 않는다.
PROFILE_CHANGELOG = {
    "v2": "손절 추가 — gross 기준. 진입 10초 만료·청산 평가 창 끝 시장가는 유지",
    "v5": "손절을 BID1 시장가로, 익절만 ASK1 지정가로 나눔",
    "v6": "트리거를 **대칭**으로 — 손절 -20bps · 익절 +20bps. 위아래를 같은 폭으로 "
          "두어 어느 쪽이 유리한지 미리 정하지 않는다",
    "v7": "진입가 -10bps 즉시 시장가 손절과 최고 BID1 대비 -10bps 추적 지정가 청산. "
          "추적 지정가는 ASK1 변경 시 새 최우선 ASK1으로 다시 게시",
    "v8": "추적 청산의 기준을 최고 BID1에서 최고 ASK1으로 변경. 현재 ASK1의 최고가 대비 "
          "-10bps 하락으로만 추적 청산을 시작",
    "v9": "실제 청산을 gross -120bps 손절, 최고 ASK1 대비 -30bps 추적 청산, 최대 900초 보유로 변경. "
          "Feature Profile의 S30 라벨은 바꾸지 않음",
    "v4": "청산을 트리거 기반으로. 진입 즉시 ASK1 게시를 없애고 gross -30/+20 에 "
          "닿으면 그때 ASK1 지정가(최대 60초). 미체결이면 BID1 시장가. "
          "평가 창 30초는 트리거를 기다리는 상한으로만 남는다",
    "v3": "결정 간격 제거 — 30초 격자에서 매 틱으로. 격자는 검증의 **측정 설계**였지 "
          "거래 규칙이 아니었다. 포지션 보유 중 차단만으로 겹침이 막히므로 간격은 "
          "기회를 버리기만 했다",
}
BACKTEST_PROTOCOL_AMBIGUOUS = "BACKTEST_PROTOCOL_AMBIGUOUS"
BACKTEST_INVARIANT_FAILURE = "BACKTEST_INVARIANT_FAILURE"
ACCOUNTING_FAILURE = "ACCOUNTING_FAILURE"

# 회계 위반 (§29~§30).
SPREAD_DOUBLE_COUNT = "SPREAD_DOUBLE_COUNT"
COST_DUPLICATION = "COST_DUPLICATION"
UNKNOWN_LEDGER_STATUS = "UNKNOWN_LEDGER_STATUS"

# 고칠 필요가 보이면 조용히 고치지 않는다 (§58).
CANONICAL_BACKTEST_SPEC_CHANGE_REQUIRED = "CANONICAL_BACKTEST_SPEC_CHANGE_REQUIRED"

# 스프레드를 어디서 무는가. 두 번 물지 않기 위해 이름으로 못 박는다.
IMPLICIT_IN_FILL_PRICE = "IMPLICIT_IN_FILL_PRICE"


# ---- 1. 정본 프로필 (§48) -------------------------------------------------------

def profile() -> dict[str, Any]:
    """이 저장소가 쓰는 유일한 실행 규칙. 값은 전부 여기 모인다.

    이전에는 이 값들이 `config.py` 와 `fill.py` 에 상수로 흩어져 있었고, 계약이 선언한
    실행 의미와 런타임 설정이 서로를 몰랐다. 그래서 인자 하나를 빠뜨리면 조용히 다른
    전략이 돌았다.
    """
    body = {
        "schema": SCHEMA_VERSION,
        "profile_id": PROFILE_ID,
        "decision": {
            "basis": "EVERY_TICK",
            "interval_seconds": None,
            "why": "백테스트는 실제로 무엇을 할 수 있었는지를 묻는다. 계속 보다가 조건이 "
                   "맞고 포지션이 없으면 산다. 간격을 두면 잡을 수 있었던 기회를 버린다",
            "blocked_while_holding": True,
            "why_blocked": "포지션을 들고 있는 동안의 신호 틱은 새 결정이 아니다. "
                           "그것을 분모에 넣으면 성과가 신호 지속시간에 좌우된다. "
                           "겹침 방지는 이것으로 충분하다 — 간격은 필요 없다",
            "not_the_validation_grid":
                "검증은 30초 격자로 anchor 를 뽑았다. 그것은 **창이 겹치지 않게 통계를 "
                "재려는** 측정 설계지 거래 규칙이 아니다. 백테스트에 그대로 가져오면 "
                "기회의 70% 를 이유 없이 버린다",
        },
        "entry": {
            "model": "BEST_QUOTE_QUEUE",
            "post_price": "BID1",
            "max_rest_seconds": cfg.ENTRY_ORDER_MAX_SECONDS,
            "max_rest_ticks": cfg.ENTRY_ORDER_MAX_TICKS,
            "timeout_basis": "CLOCK",
            "timeout_note": "시계가 먼저다. 틱 상한은 안전장치일 뿐 — 둘이 충돌하면 "
                            "먼저 닿는 쪽에서 끝난다",
        },
        "exit": {
            "model": "BEST_QUOTE_QUEUE",
            "post_price": "ASK1",
            "stop_gross_bps": cfg.STOP_GROSS_BPS,
            "trailing_drawdown_gross_bps": cfg.TRAILING_DRAWDOWN_GROSS_BPS,
            "stop_basis": "GROSS",
            "trailing_mark": "RUNNING_HIGHEST_ASK1_SINCE_ENTRY",
            "stop_action": "그 자리 **BID1 시장가**. 무조건 나간다",
            "trailing_action": (
                f"최고 ASK1 대비 {cfg.TRAILING_DRAWDOWN_GROSS_BPS:g}bps 에서 ASK1 지정가를 건다"),
            "reprice_on_best_ask_change": True,
            "reprice_action": "기존 ASK1 지정가를 취소하고 새 ASK1에 다시 게시",
            "why_stop_is_market":
                "손절은 나가는 것이 목적이다. 지정가로 걸면 '나가고 싶은데 좋은 "
                "가격에만' 이 되어 시장이 도망가는 순간에 못 나간다",
            "limit_rest_seconds": cfg.EXIT_LIMIT_REST_SECONDS,
            "limit_timeout_action": "BID1 시장가",
            "posted_at_entry": False,
            "why_not_posted_at_entry":
                "진입 즉시 ASK1 에 걸면 목표 수익이 '그 순간 스프레드' 로 고정된다. "
                "그건 방향 예측 가설이 아니라 스프레드 포획이다. 들고 있다가 가격이 "
                "실제로 움직였을 때 나간다",
            "why_gross": "net 으로 잡으면 BID1 에 사는 순간 이미 -수수료 에서 시작해 "
                         "여유가 그만큼 줄어든다. 손절선은 가격 이동을 재는 값이다",
            "stop_vs_trailing": (
                f"진입가 {cfg.STOP_GROSS_BPS:g}bps 시장가 손절은 대기 중인 추적 지정가보다도 우선한다"),
            "max_holding_seconds": cfg.EXIT_MAX_HOLD_SECONDS,
            "no_trigger_fallback": "최대 보유 시간 끝 BID1 시장가",
            "why_fallback": "안 팔린 거래를 통계에서 빼면 나쁜 거래만 사라진다",
            "note": (
                f"실제 청산은 최대 {cfg.EXIT_MAX_HOLD_SECONDS:g}초까지 손절·추적을 기다린다. "
                "S30은 Feature Profile의 미래 라벨 창이다"),
        },
        "queue": {
            "hold_through": not fill.CANCEL_ON_PRICE_CHANGE,
            "cancel_on_price_change": fill.CANCEL_ON_PRICE_CHANGE,
            "why_hold_through": "가격이 불리해질 때 주문을 빼면 역선택을 공짜로 피하게 "
                                "되어 net 이 부풀어 오른다",
            "order_qty": fill.DEFAULT_ORDER_QTY,
            "queue_fraction": fill.DEFAULT_QFRAC,
            "unexplained_fill_fraction": fill.DEFAULT_UNEXPLAINED_FILL_FRACTION,
            "price_move_alone_does_not_fill": True,
            "limits": [
                "주문 추가·취소·수정이 이 데이터에 없다. 잔량 감소를 체결과 취소로 "
                "직접 가를 수 없다",
                "큐 순번을 관측할 수 없다. 게시 시점 최우선 잔량 뒤에 선다고 가정한다",
                "정확한 거래소 모사가 아니라 보수적이고 일관된 근사다",
            ],
        },
        "cost": {
            "spread_mode": IMPLICIT_IN_FILL_PRICE,
            "spread_deduction_bps": 0.0,
            "why": "진입·청산 체결가격이 실제 BID/ASK 에서 나므로 스프레드는 이미 "
                   "손익에 들어 있다. 또 빼면 두 번 무는 것이다",
            "explicit_cost_bps": cfg.FEE_BPS,
            "explicit_cost_note": "왕복 수수료·세금. 체결가격에 들어 있지 않은 것만",
        },
        "horizon": {
            "primary_seconds": cfg.PRIMARY_HORIZON_SECONDS,
            "note": "성과를 재는 창이다. 청산 규칙이 아니다",
        },
        "not_searchable": [
            "큐 모델", "체결 가정", "비용", "결정 규칙", "진입 주문 정책",
            "청산 주문 정책", "주문 유지시간", "미설명 잔량 체결 비율",
        ],
    }
    body["profile_sha256"] = sha256_json(body)[:16]
    return body


CANONICAL = profile()


def v8_900_comparison_profile() -> dict[str, Any]:
    """V9가 고른 동일 entry를 재생할 고정 V8-900 청산 비교 프로필.

    과거 V8의 -10bp 손절·-10bp 추적선은 유지하되, V9와 같은 최대 보유 900초를 쓴다.
    그래서 보유 시간 차이가 아니라 청산 조건 차이를 읽을 수 있다. V9가
    Parameter Search에서 entry·q를 고르고, V8-900은 Validation과 Final의 청산
    비교에만 쓴다. 후보 승격 권한은 정본 V9에만 있다.
    """
    comparison = profile()
    comparison["schema"] = "canonical_backtest_profile.v8_900_comparison"
    comparison["profile_id"] = V8_900_COMPARISON_PROFILE_ID
    comparison["exit"] = {
        **comparison["exit"],
        "stop_gross_bps": -10.0,
        "trailing_drawdown_gross_bps": -10.0,
        "trailing_action": "최고 ASK1 대비 -10bps 에서 ASK1 지정가를 건다",
        "stop_vs_trailing": "진입가 -10bps 시장가 손절은 대기 중인 추적 지정가보다도 우선한다",
        "max_holding_seconds": cfg.EXIT_MAX_HOLD_SECONDS,
        "no_trigger_fallback": "최대 보유 시간 끝 BID1 시장가",
        "note": (
            f"V8의 -10bp 손절·추적선을 유지하고 최대 {cfg.EXIT_MAX_HOLD_SECONDS:g}초까지 "
            "감시하는 비교 전용 프로필이다. V9가 entry·q를 고른 뒤에만 별도 재생한다"),
    }
    comparison["comparison_only"] = {
        "selection_profile_id": PROFILE_ID,
        "selection_profile_sha256": CANONICAL["profile_sha256"],
        "role": "V9가 entry·q와 승격을 결정하고, V8-900은 청산 비교에만 쓴다",
        "promotion_authority": False,
    }
    comparison.pop("profile_sha256", None)
    comparison["profile_sha256"] = sha256_json(comparison)[:16]
    return comparison


V8_900_COMPARISON = v8_900_comparison_profile()


def profile_hash() -> str:
    return CANONICAL["profile_sha256"]


# ---- 2. 결정 기회 (§13~§14) -----------------------------------------------------

def decision_grid(time_s: np.ndarray, interval_seconds: float) -> np.ndarray:
    """시계 격자 위의 틱만 True. 신호가 켜져도 격자 밖이면 결정이 아니다.

    검증이 이 표본 위에서 성립했다. 매 틱 진입으로 바꾸면 기회 수와 상관 구조가 함께
    달라져 — 그건 성능 개선이 아니라 다른 실험이다.
    """
    time_s = np.asarray(time_s, dtype=float)
    if not len(time_s):
        return np.zeros(0, dtype=bool)
    edges = np.arange(time_s[0], time_s[-1] + interval_seconds, interval_seconds)
    ticks = np.searchsorted(time_s, edges, side="left")
    ticks = ticks[ticks < len(time_s)]
    mask = np.zeros(len(time_s), dtype=bool)
    mask[np.unique(ticks)] = True
    return mask


# ---- 3. 계약과 프로필이 어긋나면 거부한다 (§22~§25) -----------------------------

def check_contract(contract_template: Mapping[str, Any],
                   canonical: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """계약이 실행 방식을 스스로 정하려 들면 막는다.

    계약이 담당하는 것은 **신호와 방향**이다. 실행은 정본이 담당한다. 계약에 실행
    결속이 적혀 있어도 정본과 다르면 조용히 넘어가지 않는다.
    """
    canonical = CANONICAL if canonical is None else canonical
    binding = contract_template.get("execution_binding") or {}
    problems: list[str] = []
    if not binding:
        return {"ok": True, "problems": [], "declared": {}, "note": "계약이 실행을 "
                "선언하지 않는다. 정본이 정한다"}

    checks = [
        ("decision_opportunity", canonical["decision"]["basis"],
         binding.get("decision_opportunity")),
        ("decision_spacing_seconds", canonical["decision"]["interval_seconds"],
         binding.get("decision_spacing_seconds")),
        ("evaluation_horizon", f"S{int(canonical['horizon']['primary_seconds'])}",
         binding.get("evaluation_horizon")),
    ]
    for name, expected, actual in checks:
        if actual is not None and actual != expected:
            problems.append(f"{name}: 계약이 {actual!r} 라 했는데 정본은 {expected!r} 다")

    mode = binding.get("entry_execution")
    if mode and mode != canonical["entry"]["model"]:
        problems.append(
            f"entry_execution: 계약이 {mode!r} 라 했는데 정본은 "
            f"{canonical['entry']['model']!r} 다. 실행 방식은 가설이 정하지 않는다")
    return {"ok": not problems, "problems": problems, "declared": dict(binding),
            "profile_sha256": canonical["profile_sha256"]}


# ---- 4. 실행 (§10, §24) ---------------------------------------------------------

def run_backtest(contracts: Mapping[str, Mapping[str, Any]],
                 members: Mapping[str, Sequence[str]], dates: Sequence[str],
                 output: Path, *, canonical: Mapping[str, Any] | None = None,
                 parameter_table: Mapping[str, Mapping[str, float]] | None = None,
                 workers: int = 8, root: Path = cfg.TICK_ROOT,
                 reference_dates: Sequence[str] = (),
                 strict_contract: bool = True) -> dict[str, Any]:
    """정본 규칙으로 돌린다. 실행 설정을 인자로 받지 않는다 — 프로필이 정한다."""
    canonical = CANONICAL if canonical is None else canonical
    if strict_contract:
        for name, template in contracts.items():
            check = check_contract(template, canonical)
            if not check["ok"]:
                raise RuntimeError(
                    f"{CANONICAL_BACKTEST_SPEC_CHANGE_REQUIRED}: 계약 {name} 이 정본과 "
                    "다른 실행을 요구한다. 조용히 맞추지 않는다:\n  "
                    + "\n  ".join(check["problems"]))

    manifest = ledger.build_ledger(
        contracts=contracts, members=members, dates=dates, output=Path(output),
        workers=workers, root=root,
        # 아래는 전부 프로필에서 온다. 부르는 쪽이 고를 수 없다.
        limit_entry=canonical["entry"]["model"] == "BEST_QUOTE_QUEUE",
        limit_exit=canonical["exit"]["model"] == "BEST_QUOTE_QUEUE",
        decision_grid_seconds=canonical["decision"]["interval_seconds"],  # None = 매 틱
        entry_max_rest_seconds=canonical["entry"].get("max_rest_seconds", cfg.ENTRY_ORDER_MAX_SECONDS),
        entry_max_rest_ticks=canonical["entry"].get("max_rest_ticks", cfg.ENTRY_ORDER_MAX_TICKS),
        stop_gross_bps=canonical["exit"].get("stop_gross_bps"),
        trailing_drawdown_gross_bps=canonical["exit"].get("trailing_drawdown_gross_bps"),
        exit_limit_rest_seconds=canonical["exit"].get("limit_rest_seconds"),
        horizon_seconds=canonical["exit"].get("max_holding_seconds"),
        parameter_table=parameter_table, reference_dates=reference_dates)
    manifest["canonical_backtest_profile_id"] = canonical["profile_id"]
    manifest["canonical_backtest_profile_hash"] = canonical["profile_sha256"]
    manifest["contract_sha256"] = {
        name: contract_runtime.contract_hash(template)
        for name, template in sorted(contracts.items())
    }
    manifest["contracts_sha256"] = sha256_json(manifest["contract_sha256"])
    write_json(Path(output).with_name(Path(output).stem + "_manifest.json"), manifest)
    return manifest


# ---- 5. 회계 불변식 (§29~§30) ---------------------------------------------------

def spread_accounting_audit(frame: pd.DataFrame, manifest: Mapping[str, Any],
                            canonical: Mapping[str, Any] | None = None
                            ) -> dict[str, Any]:
    """스프레드를 두 번 물지 않았는가. 비용은 정확히 한 번인가.

    체결가격이 실제 호가에서 났는데 `spread_bps` 를 또 빼면 같은 비용을 두 번 무는
    것이다. 그러면 멀쩡한 신호도 전부 손실로 보인다.
    """
    canonical = CANONICAL if canonical is None else canonical
    problems: list[str] = []
    fee = float(canonical["cost"]["explicit_cost_bps"])

    if canonical["cost"]["spread_mode"] != IMPLICIT_IN_FILL_PRICE:
        problems.append(f"{SPREAD_DOUBLE_COUNT}: 스프레드 처리 방식이 정본과 다르다")
    if float(canonical["cost"].get("spread_deduction_bps", 0.0)) != 0.0:
        problems.append(f"{SPREAD_DOUBLE_COUNT}: 체결가격을 쓰면서 스프레드를 또 뺀다")

    filled = frame.loc[frame["status"].eq(ledger.FILLED)] if len(frame) else frame
    identity_gap = None
    if len(filled) and {"gross_bps", "net_bps"} <= set(filled.columns):
        gap = (filled["gross_bps"].to_numpy(float)
               - filled["net_bps"].to_numpy(float))
        identity_gap = float(np.nanmax(np.abs(gap - fee))) if len(gap) else 0.0
        if identity_gap > 1e-6:
            problems.append(
                f"{COST_DUPLICATION}: gross - net 이 수수료 {fee} 와 다르다 "
                f"(최대 어긋남 {identity_gap:.6f} bps). 비용이 두 번 빠졌거나 "
                "다른 값이 섞였다")
    return {"schema": "backtest_accounting_audit.v1", "created_at": now_utc(),
            "ok": not problems, "problems": problems,
            "spread_mode": canonical["cost"]["spread_mode"],
            "spread_deduction_bps": canonical["cost"]["spread_deduction_bps"],
            "explicit_cost_bps": fee,
            "gross_minus_net_max_gap_bps": identity_gap,
            "checked_fills": int(len(filled)),
            "why": "체결가격이 실제 BID/ASK 에서 나므로 스프레드는 이미 손익에 있다. "
                   "따로 빼는 것은 중복이다"}


# ---- 6. 지표 분모 (§36~§38) -----------------------------------------------------

METRIC_DENOMINATORS = {
    "bps_per_decision": "채점 가능한 결정 (scorable = FILLED + UNFILLED)",
    "bps_per_fill": "실제 체결",
    "bps_per_day": "거래일",
    "fill_rate": "결정 대비 체결",
    "signal_rate": "전체 틱 대비 신호 틱 (진단용)",
}
# 분모가 무엇인지 이름이 말해주지 않는 것. 쓰지 않는다.
FORBIDDEN_METRIC_NAMES = ("bps_per_attempt", "attempts", "attempt_rate")


def block_summary(frame: pd.DataFrame) -> dict[str, Any] | None:
    """어느 종목-일에 걸렸는가. 총합 하나로는 안 보이는 것을 본다.

    `net_bps_total` 은 합이라 블록 하나가 전부를 만들어도 같은 값이 나온다.

    쏠림 자체는 결함이 아니다. 같은 수식을 전 종목에 똑같이 적용했으므로, 쏠렸다는
    것은 종목을 고른 것이 아니라 **수식이 잡은 구조**다. 종목별로 파라미터를 맞췄다면
    다른 얘기지만 이 Workflow 는 그렇게 하지 않는다.

    그래서 이 값들은 경고가 아니라 서술이다. 쏠린 종목이 가설이 말한 성격을 갖는지는
    원장의 `day_spread_bps_median`·`day_tick_size_bps_median`·`day_trade_tick_count`
    로 대조한다. 선택 기준은 바꾸지 않는다.
    """
    if not len(frame) or "net_bps" not in frame.columns:
        return None
    filled = (frame.loc[frame["status"].eq(ledger.FILLED)]
              if "status" in frame.columns else frame)
    net = pd.to_numeric(filled.get("net_bps"), errors="coerce")
    net = net.loc[net.notna()]
    if not len(net):
        return None
    rows = filled.loc[net.index]
    blocks = net.groupby([rows["symbol"], rows["date"]]).sum()
    by_symbol = net.groupby(rows["symbol"]).sum()
    absolute = by_symbol.abs().sum()
    top = by_symbol.abs().idxmax()
    return {
        "definition": ("블록 = (종목, 날). 체결 행의 net_bps 합. "
                       "쏠림은 결함이 아니라 수식이 잡은 구조다 — "
                       "원장의 day_* 열로 그 종목의 성격과 대조한다"),
        "count": int(len(blocks)),
        "positive_share": float((blocks > 0.0).mean()),
        "net_median": float(blocks.median()),
        # 블록이 3개 이하면 위 3개를 빼면 아무것도 안 남아 항상 0 이 된다.
        # 그 0 은 "상위를 빼도 0 이다" 로 읽히므로 아예 내지 않는다.
        "net_total_excluding_top3": (float(blocks.sum() - blocks.nlargest(3).sum())
                                     if len(blocks) > 3 else None),
        "top_symbol": str(top),
        "top_symbol_net": float(by_symbol.loc[top]),
        "top_symbol_abs_share": (float(abs(by_symbol.loc[top]) / absolute)
                                 if absolute > 0.0 else None),
    }


def metric_report(frame: pd.DataFrame, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """하나만 보고 고르지 않도록 분모가 다른 수를 나란히 낸다."""
    measured = metrics.measure(frame, manifest)
    return {"schema": "backtest_metrics.v1", "created_at": now_utc(),
            "eligible_decisions": measured.decisions,
            "scorable": measured.scorable,
            "orders": measured.decisions,
            "fills": measured.fills,
            "unfilled": measured.unfilled,
            "censored": measured.censored,
            "blocked_ticks": measured.blocked_ticks,
            "signal_ticks": measured.signal_ticks,
            "fill_rate": (measured.fills / measured.decisions
                          if measured.decisions else None),
            "net_bps_total": measured.total_net_bps,
            "gross_bps_total": measured.total_gross_bps,
            "net_bps_per_decision": measured.bps_per_decision,
            "net_bps_per_fill": measured.bps_per_fill,
            "net_bps_per_day": measured.bps_per_day,
            "blocks": block_summary(frame),
            "denominators": METRIC_DENOMINATORS,
            "note": "체결률이 후보마다 달라질 수 있다. 손익이 좋아졌을 때 신호가 좋아진 "
                    "것인지 체결이 골라진 것인지 구분하려면 둘을 함께 봐야 한다"}


# ---- 7. 숨은 상수 목록 (§17~§18) ------------------------------------------------

def hidden_constants(canonical: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """결과를 바꾸는데 계약 밖에 있던 값들. 어느 것이 프로필로 옮겨졌는지 보인다."""
    canonical = CANONICAL if canonical is None else canonical
    return [
        {"constant": "LIMIT_ENTRY", "was": cfg.LIMIT_ENTRY, "위치": "config.py",
         "이제": f"profile.entry.model = {canonical['entry']['model']}"},
        {"constant": "LIMIT_EXIT", "was": cfg.LIMIT_EXIT, "위치": "config.py",
         "이제": f"profile.exit.model = {canonical['exit']['model']}"},
        {"constant": "ENTRY_ORDER_MAX_SECONDS", "was": cfg.ENTRY_ORDER_MAX_SECONDS,
         "위치": "config.py", "이제": "profile.entry.max_rest_seconds"},
        {"constant": "ENTRY_ORDER_MAX_TICKS", "was": cfg.ENTRY_ORDER_MAX_TICKS,
         "위치": "config.py", "이제": "profile.entry.max_rest_ticks (안전장치)"},
        {"constant": "EXIT_ORDER_RESTS_TO_HORIZON", "was": cfg.EXIT_ORDER_RESTS_TO_HORIZON,
         "위치": "config.py", "이제": "profile.exit.rests_to_horizon"},
        {"constant": "FEE_BPS", "was": cfg.FEE_BPS, "위치": "config.py",
         "이제": "profile.cost.explicit_cost_bps"},
        {"constant": "PRIMARY_HORIZON_SECONDS", "was": cfg.PRIMARY_HORIZON_SECONDS,
         "위치": "config.py", "이제": "profile.horizon.primary_seconds"},
        {"constant": "DEFAULT_QFRAC", "was": fill.DEFAULT_QFRAC, "위치": "fill.py",
         "이제": "profile.queue.queue_fraction"},
        {"constant": "DEFAULT_UNEXPLAINED_FILL_FRACTION",
         "was": fill.DEFAULT_UNEXPLAINED_FILL_FRACTION, "위치": "fill.py",
         "이제": "profile.queue.unexplained_fill_fraction"},
        {"constant": "CANCEL_ON_PRICE_CHANGE", "was": fill.CANCEL_ON_PRICE_CHANGE,
         "위치": "fill.py", "이제": "profile.queue.cancel_on_price_change"},
        {"constant": "decision cadence", "was": "매 신호 틱 (flat 이면)",
         "위치": "ledger.py 암묵",
         "이제": f"profile.decision.basis = {PROFILE_ID and 'EVERY_TICK'} — 명시적으로 매 틱"},
    ]


# ---- 8. 관문 (§53~§54) ----------------------------------------------------------

def readiness_gate(accounting: Mapping[str, Any], invariants_ok: bool,
                   tests_ok: bool, contract_check: Mapping[str, Any] | None = None,
                   canonical: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """다섯 조건을 모두 만족해야 이 장비 위에서 후보를 비교할 수 있다."""
    canonical = CANONICAL if canonical is None else canonical
    profile_defined = bool(canonical.get("profile_sha256"))
    contract_ok = True if contract_check is None else bool(contract_check.get("ok"))
    denominators_ok = all(name not in METRIC_DENOMINATORS
                          for name in FORBIDDEN_METRIC_NAMES)

    if not profile_defined or not contract_ok:
        status, why = BACKTEST_PROTOCOL_AMBIGUOUS, "정본 프로필이 없거나 계약과 어긋난다"
    elif not accounting.get("ok"):
        status, why = ACCOUNTING_FAILURE, "스프레드·비용 회계가 맞지 않는다"
    elif not invariants_ok or not tests_ok:
        status, why = BACKTEST_INVARIANT_FAILURE, "원장 불변식 또는 테스트가 통과하지 않는다"
    else:
        status, why = READY_FOR_CANONICAL_BACKTEST_SEARCH, "다섯 조건을 모두 만족한다"

    return {"status": status, "why": why,
            "CANONICAL_BACKTEST_PROFILE_DEFINED": profile_defined,
            "BACKTEST_TESTS_PASS": bool(tests_ok),
            "LEDGER_INVARIANTS_PASS": bool(invariants_ok),
            "SPREAD_ACCOUNTING_PASS": bool(accounting.get("ok")),
            "METRIC_DENOMINATORS_PASS": denominators_ok,
            "CONTRACT_MATCHES_PROFILE": contract_ok,
            "profile_sha256": canonical["profile_sha256"],
            "note": "프로필 해시가 다르면 같은 평가 환경에서 비교한 것이 아니다"}
