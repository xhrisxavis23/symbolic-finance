"""이 연구 환경에서 무엇을 볼 수 있고 무엇을 볼 수 없는가.

Agent 에게 데이터 감사 문서를 통째로 던지지 않는다. 대신 여기서 **결정적으로** 만든
한 장짜리 프로필을 준다.

## 이 프로필의 목적은 상상을 줄이는 것이 아니다

    데이터 환경은 **무엇을 관측했다고 말할 수 있는지**를 정한다.
    **어떤 금융 메커니즘을 생각해도 되는지**를 정하지 않는다.

직접 관측되지 않는 메커니즘을 금지하면 "feature A 높고 B 낮으면 오른다" 수준의 얕은
상관 가설만 남는다. 그래서 메커니즘은 열어 두되, 그것을 **관측 사실로 말하는 것**과
**검증 가능한 예측 없이 주장하는 것**을 막는다.

## 세 가지를 섞지 않는다

    연구 capability   participant ID 가 없다        → Agent 에게 준다
    측정 한계         스냅샷 간격이 종목마다 다르다  → Agent 에게 준다
    데이터 위생       특정 날짜 파일 문제            → 코드가 처리한다. Agent 는 모른다
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import catalog, data as tickdata
from .config import PRIMARY_HORIZON_SECONDS, now_utc


# 관측 가능성 등급. 하나의 어휘만 쓴다.
DIRECT = "DIRECT"                    # 원시 또는 정본 feature 에서 바로
DERIVABLE = "DERIVABLE"              # 현재 데이터와 카탈로그 연산자로 계산
PROXY_ONLY = "PROXY_ONLY"            # 원 개념은 못 보고 제한된 대용치만
VALIDATION_ONLY = "VALIDATION_ONLY"  # 데이터는 있으나 가설 생성 시 접근 금지
UNAVAILABLE = "UNAVAILABLE"          # 현재 데이터에 없다
STATUSES = (DIRECT, DERIVABLE, PROXY_ONLY, VALIDATION_ONLY, UNAVAILABLE)


# `data.load()` 가 실제로 내놓는 것에서 유도한다. 못 만드는 것을 만들 수 있다고 적으면
# 그 순간 프로필이 거짓이 된다.
CAPABILITIES: dict[str, dict[str, str]] = {
    # -- 호가 --------------------------------------------------------------
    "l1_quote": {"status": DIRECT, "group": "book",
                 "evidence": "bid_price/ask_price 의 최우선 열"},
    "multi_level_depth": {"status": DIRECT, "group": "book",
                          "evidence": f"L1~L{tickdata.N_LEVELS} 가격·잔량"},
    "spread": {"status": DERIVABLE, "group": "book",
               "evidence": "`spread_bps` — ASK1-BID1"},
    "book_imbalance": {"status": DERIVABLE, "group": "book",
                       "evidence": "`book_imbalance` — L1 잔량 비대칭"},
    "depth_aggregation": {"status": DERIVABLE, "group": "book",
                          "evidence": "`level_aggregate` 연산자로 L1~L10 합산"},
    "depth_beyond_l10": {"status": UNAVAILABLE, "group": "book",
                         "evidence": f"게시 깊이가 L{tickdata.N_LEVELS} 까지만 기록된다"},
    "hidden_liquidity": {"status": UNAVAILABLE, "group": "book",
                         "evidence": "스냅샷은 게시분만 담는다. 빙산·미표시 물량은 없다"},
    # -- 체결 --------------------------------------------------------------
    "trade_prints": {"status": DIRECT, "group": "trade",
                     "evidence": "buy_volume/sell_volume — `askbid_type` 으로 방향 구분"},
    "aggressor_direction": {"status": DIRECT, "group": "trade",
                            "evidence": "`askbid_type` 으로 매수·매도 주도 구분"},
    "signed_flow": {"status": DERIVABLE, "group": "trade",
                    "evidence": "`signed_aggr_flow_20/100` — 방향별 체결량 비"},
    "trade_extreme_prices": {"status": DIRECT, "group": "trade",
                             "evidence": "buy_max_price/sell_min_price"},
    # -- 주문 이벤트 --------------------------------------------------------
    "order_submission": {"status": UNAVAILABLE, "group": "order_event",
                         "evidence": "주문 이벤트 로그가 아니라 스냅샷 데이터다"},
    "order_cancellation": {"status": UNAVAILABLE, "group": "order_event",
                           "evidence": "위와 같다"},
    "order_modification": {"status": UNAVAILABLE, "group": "order_event",
                           "evidence": "위와 같다"},
    "queue_position": {"status": UNAVAILABLE, "group": "order_event",
                       "evidence": "주문 ID·큐 순번이 없다"},
    "participant_identity": {"status": UNAVAILABLE, "group": "order_event",
                             "evidence": "데이터에 없다"},
    "order_add_cancel_decomposition": {
        "status": PROXY_ONLY, "group": "order_event",
        "evidence": "잔량 감소에서 체결분을 뺀 나머지로만 추정한다",
        "limitation": "스냅샷 간격 안에서 일어난 추가와 취소가 상쇄되면 가를 수 없다"},
    "quote_repricing": {"status": DERIVABLE, "group": "order_event",
                        "evidence": "최우선호가 변화 — `*_best_price_drop_count_50t`"},
    "displayed_depth_change": {"status": DERIVABLE, "group": "order_event",
                               "evidence": "레벨별 잔량 스냅샷 차분"},
    # -- 외부 --------------------------------------------------------------
    "information_arrival": {"status": UNAVAILABLE, "group": "external",
                            "evidence": "뉴스·공시 등 외부 사건이 이 데이터셋에 없다"},
    "pre_anchor_volatility": {"status": UNAVAILABLE, "group": "external",
                              "evidence": "카탈로그에 사전 변동성 정본 feature 가 없다"},
    # -- 시간 --------------------------------------------------------------
    "pre_anchor_temporal_grid": {"status": DIRECT, "group": "time",
                                 "evidence": "Profiler 의 -10s/-5s/-2s/-1s/0s 격자"},
    "post_anchor_executable_quotes": {
        "status": VALIDATION_ONLY, "group": "time",
        "evidence": "원본에 있다. 검증 단계에서 예측을 시험할 때만 연다",
        "limitation": "가설을 쓸 때 보면 자기 답을 보고 쓰는 것이 된다"},
    "discovery_price_path": {
        "status": DIRECT, "group": "time",
        "evidence": "Feature Profile에 포함된 discovery anchor의 BID1/ASK1 경로를 읽기 전용 tool로 조회",
        "limitation": "미래 최고 ASK1을 포함하는 oracle discovery 자료다. 체결·검증·OOS 성과로 해석하지 않는다"},
}


# 실측한 측정 한계. 손으로 적지 말고 다시 재서 갈아끼운다.
# 측정: 2026-08-20 · 코어 28종목 중 앞 8개 × 20260317 정규장.
MEASURED = {
    "measured_at": "2026-08-20",
    "sample": "core 28 symbols 중 앞 8개 × 20260317",
    "median_quote_interval_seconds": {"min": 0.315, "max": 4.942, "spread_ratio": 15.7},
    "note": "종목 간 호가 간격이 15.7배까지 벌어진다. 그래서 `100 ticks` 같은 표현을 "
            "종목 사이에서 같은 금융 시간으로 읽으면 안 된다",
}


# 데이터 위생 — 코드가 처리한다. Agent 에게는 이런 것이 있었다는 사실도 알리지 않는다 (§9).
# `data.available_dates()` 는 폴더 이름이 숫자인지만 본다. 이 날짜들은 실거래일이 아니다.
NOT_REAL_DATES = {"20260427": "3종목뿐이고 전부 *_synthesized_*. 실데이터가 아니다"}


def dataset_coverage(root=None) -> dict[str, Any]:
    """실제로 읽을 수 있는 **실거래일**. 파일에서 센다 — 손으로 적지 않는다.

    합성 날짜를 세면 프로필이 그 자리에서 거짓이 된다.
    """
    dates = tickdata.available_dates() if root is None else tickdata.available_dates(root)
    dates = [d for d in dates if d not in NOT_REAL_DATES]
    return {"trading_days": len(dates),
            "first_date": dates[0] if dates else None,
            "last_date": dates[-1] if dates else None,
            "regime_diversity": "LIMITED",
            "note": "단일 구간의 짧은 표본이다. \"여러 시장 국면에 걸쳐\" 같은 일반화를 "
                    "이 데이터로 주장할 수 없다"}


def research_capability_profile(consumed_dates: Sequence[str] = (),
                                root=None) -> dict[str, Any]:
    """Agent 에게 줄 환경 정의. 결정적이다 — 같은 저장소면 같은 값이 나온다."""
    coverage = dataset_coverage(root)
    groups: dict[str, list[dict[str, str]]] = {}
    for name, item in sorted(CAPABILITIES.items()):
        groups.setdefault(item["group"], []).append(
            {"capability": name, **{k: v for k, v in item.items() if k != "group"}})
    return {
        "schema": "research_capability_profile.v1",
        "created_at": now_utc(),
        "catalog_sha256": catalog.catalog_hash(),
        "policy": (
            "이 환경 정의는 **무엇을 관측했다고 말할 수 있는지**를 정한다. "
            "**어떤 금융 메커니즘을 생각해도 되는지**를 정하지 않는다."),
        "data_type": "quote snapshots + trade records (NOT a full order-event log)",
        "statuses": {
            DIRECT: "원시 또는 정본 feature 에서 바로 관측된다",
            DERIVABLE: "현재 데이터와 카탈로그 연산자로 계산된다",
            PROXY_ONLY: "원 개념은 직접 관측되지 않고 제한된 대용치만 있다",
            VALIDATION_ONLY: "데이터는 있으나 가설을 쓸 때는 열지 않는다. "
                             "예측을 시험하는 데 쓴다",
            UNAVAILABLE: "현재 데이터에 없다"},
        "capabilities": groups,
        "measurement_limits": MEASURED,
        "max_observable_depth": tickdata.N_LEVELS,
        "dataset_coverage": coverage,
        "outcome_horizon_seconds": float(PRIMARY_HORIZON_SECONDS),
        "consumed_dates": list(consumed_dates),
        "consumed_note": "이미 결과를 본 날짜다. 수정된 가설을 여기서 다시 확증할 수 없다",
        "catalog_features": sorted(
            catalog.default_policy().grounding_visible
            - catalog.default_policy().execution_only),
    }


def experiment_environment(*, dates: Mapping[str, Sequence[str]],
                           symbols: Sequence[str] | None = None,
                           symbol_group: str | None = None,
                           selection_rule: str | None = None,
                           known_limits: Sequence[str] | None = None) -> dict[str, Any]:
    """이 실험이 **무엇 위에서** 돌았는가. 보고서 맨 앞에 고정으로 붙인다.

    손으로 적으면 어긋난다. 종목군·종목·기간을 코드가 만들어야 나중에 "24 종목" 같은
    재현 불가능한 기록이 남지 않는다.

    `dates` 는 구간 이름 → 날짜 목록이다. 예: {"검증": [...], "발굴": [...]}.
    """
    from . import universe

    symbols = list(symbols if symbols is not None else catalog.DEFAULT_PROFILE_SYMBOLS)
    groups = universe.members()
    used = set(symbols)
    rows = []
    for key, cluster in sorted(universe.CLUSTERS.items()):
        core = groups.get(key, [])
        picked = [s for s in core if s in used]
        rows.append({"cluster": key, "label": cluster.label,
                     "universe_size": cluster.universe_size,
                     "core_size": len(core), "used": picked,
                     "spread_bps": cluster.spread_bps})
    unclustered = sorted(used - {s for group in groups.values() for s in group})
    core_total = sum(len(g) for g in groups.values())
    coverage = dataset_coverage()
    core_sizes = sorted({row["core_size"] for row in rows})


    profile = catalog.DEFAULT_PROFILE
    return {
        "schema": "experiment_environment.v1",
        "market": profile.market,
        "symbol_group": symbol_group or (
            f"미시구조 클러스터 {len(universe.CLUSTERS)}개 (MK01~MK07) "
            f"코어 {core_total}종목 중 클러스터당 앞 {len(symbols) // len(rows)}개"),
        "selection_rule": selection_rule or (
            "각 클러스터 명단 앞에서 자른다. 무작위도 대표성 표집도 아니다 — "
            "재현 가능한 선택인 것이 이 방식의 장점이다"),
        "symbol_count": len(symbols),
        "symbols": symbols,
        "clusters": rows,
        "unclustered_symbols": unclustered,
        "core_universe_size": core_total,
        "coverage_ratio": len(symbols) / core_total if core_total else None,
        "dates": {name: list(group) for name, group in dates.items()},
        "session": session_window(),
        "profile_id": profile.profile_id,
        "profile_dates": list(profile.date_range),
        "catalog_sha256": catalog.catalog_hash(),
        "members_sha256": universe.members_sha256()[:16],
        "known_limits": list(known_limits) if known_limits is not None else [
            f"프로필 측정과 가설 검증이 **같은 {len(symbols)}종목**을 쓴다. 나머지 "
            f"{core_total - len(symbols)}종목에서 성립하는지는 보지 않았다",
            f"클러스터 크기가 다른데({core_sizes[0]}~{core_sizes[-1]}) 똑같은 개수를 "
            "앞에서 뽑았다. 큰 클러스터가 과소대표된다",
            f"실거래일 {coverage['trading_days']}일 한 구간뿐이라 "
            "시장 국면을 건너뛴 일반화를 주장할 수 없다",
        ],
    }


def environment_markdown(env: Mapping[str, Any]) -> str:
    """보고서 맨 앞에 붙일 절. 모든 보고서가 같은 모양이어야 비교가 된다."""
    L = ["## 실험 환경", "", "| 항목 | 값 |", "|---|---|",
         f"| 시장 | {env['market']} |",
         f"| 종목군 | {env['symbol_group']} |",
         f"| 종목 수 | **{env['symbol_count']}개** "
         f"(코어 {env['core_universe_size']}종목의 {env['coverage_ratio']:.1%}) |"]
    for name, group in env["dates"].items():
        span = f"{group[0]}~{group[-1]}" if len(group) > 1 else (group[0] if group else "—")
        L.append(f"| 기간 · {name} | `{span}` ({len(group)}일) |")
    L += [f"| 세션 | {env['session']} |",
          f"| feature profile | `{env['profile_id']}` "
          f"(측정 {'~'.join(env['profile_dates'])}) |",
          f"| catalog | `{env['catalog_sha256']}` |", "",
          "### 종목", "", "| 클러스터 | 성격 | 코어 | 쓴 종목 |", "|---|---|---|---|"]
    for row in env["clusters"]:
        L.append(f"| `{row['cluster']}` | {row['label']} | {row['core_size']} | "
                 f"{', '.join(f'`{s}`' for s in row['used'])} |")
    if env["unclustered_symbols"]:
        L.append(f"| — | 클러스터 없음 | | "
                 f"{', '.join(f'`{s}`' for s in env['unclustered_symbols'])} |")
    L += ["", f"선택 규칙: {env['selection_rule']}", "", "### 이 환경의 한계", ""]
    L += [f"- {item}" for item in env["known_limits"]] + [""]
    return "\n".join(L)


def session_window() -> str:
    from . import data as td
    def clock(v: int) -> str:
        return f"{v // 10 ** 10:02d}:{v // 10 ** 8 % 100:02d}:{v // 10 ** 6 % 100:02d}"
    return f"{clock(td.SESSION_START)}~{clock(td.SESSION_END)} 정규장"


def lookup(name: str) -> str | None:
    item = CAPABILITIES.get(name)
    return item["status"] if item else None


def validate_requirements(requirements: Sequence[Mapping[str, Any]]) -> list[str]:
    """가설이 적은 capability 요구가 실제 환경과 맞는가. 코드가 본다."""
    problems: list[str] = []
    for item in requirements or []:
        if not isinstance(item, Mapping):
            problems.append(f"capability 요구가 객체가 아니다: {item!r}")
            continue
        name = str(item.get("capability", ""))
        claimed = str(item.get("availability", ""))
        actual = lookup(name)
        if actual is None:
            problems.append(f"모르는 capability '{name}'")
            continue
        if claimed not in STATUSES:
            problems.append(f"{name}: 모르는 availability '{claimed}'")
        elif claimed != actual:
            problems.append(f"{name}: '{claimed}' 라고 적었는데 실제는 '{actual}' 다")
    return problems
