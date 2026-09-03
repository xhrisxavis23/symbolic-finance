"""시장 상태·중심 종목 구성으로 Research Workflow를 반복 실행한다."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .. import data as tickdata, featureprofile, search, universe
from ..agents.runtime import Runner
from ..config import TICK_ROOT, now_utc, write_json
from ..contracts.artifacts import read_artifact
from ..modules import market_regime
from ..modules.profile import ProfileSelection
from ..modules.schedule import ResearchSchedule
from .direct_evidence_research import (DirectEvidenceResearchRequest,
                                       run as run_direct_evidence_research)
from .research import ResearchRequest, run as run_research


@dataclass(frozen=True)
class MarketExperimentRequest:
    clusters: tuple[str, ...]
    profile_store: Path
    symbol_day_labels: Path
    discovery_dates: tuple[str, ...]
    validation_dates: tuple[str, ...]
    test_dates: tuple[str, ...]
    discovery_profit_root: Path | None = None
    per_cluster: tuple[int, ...] = (5, 1)
    validation_day_count: int = 3
    test_day_count: int = 3
    max_workers: int = 1
    resume: bool = False
    retry_errors: bool = False
    direct_evidence_only: bool = False


def _schedule(labels: Any, request: MarketExperimentRequest, regime: str,
              discovery_date: str | None = None
              ) -> tuple[ResearchSchedule | None, str | None]:
    if request.validation_day_count < 1 or request.test_day_count < 1:
        raise ValueError("Validation·Test 일수는 1 이상이어야 한다")
    try:
        discovery = ((str(discovery_date),) if discovery_date is not None else
                     market_regime.choose(labels, request.discovery_dates, regime, 1))
        validation = market_regime.choose(labels, request.validation_dates, regime,
                                          request.validation_day_count)
        test = market_regime.choose(labels, request.test_dates, regime, request.test_day_count)
    except ValueError as error:
        return None, str(error)
    return ResearchSchedule(
        validation_dates=validation,
        search_fit_dates=discovery,
        search_confirm_dates=(),
        backtest_dates=test,
        terminal_oos_dates=(),
        single_day_discovery_search=True,
    ), None


def _discovery_dates(labels: Any, request: MarketExperimentRequest, regime: str) -> list[str]:
    allowed = {str(value) for value in request.discovery_dates}
    return [str(value) for value in labels.loc[
        labels["date"].isin(allowed) & labels["regime"].eq(str(regime)), "date"].tolist()]


def _search_reference(symbols: Sequence[str], date: str, *, root: Path = TICK_ROOT
                      ) -> tuple[str | None, str | None]:
    """Search가 실제로 쓸 당일 직전 100 tick이 중심 종목 모두에 있는지 확인한다."""
    reference = str(date)
    missing = []
    for symbol in symbols:
        try:
            arrays, _ = tickdata.load(str(symbol), reference, root)
        except (FileNotFoundError, tickdata.InsufficientQuoteData):
            missing.append(str(symbol))
            continue
        if len(arrays.get("time_s", ())) <= search.SearchConfig().min_calibration_obs:
            missing.append(str(symbol))
    if missing:
        return reference, f"당일 직전 100 tick을 만들 수 없는 중심 종목: {','.join(missing)}"
    return reference, None


def plan(request: MarketExperimentRequest, output: Path) -> dict[str, Any]:
    """날짜 상태와 Profile이 있는 중심 종목을 결과 보기 전에 고정한다."""
    output = Path(output)
    labels = market_regime.build(request.symbol_day_labels, output / "market_regime")
    available = {(cluster, symbol, date) for cluster, symbol, date
                 in featureprofile.available(request.profile_store)}
    units: list[dict[str, Any]] = []
    for regime in (market_regime.UP, market_regime.DOWN, market_regime.SIDEWAYS):
        # Validation/Test의 상태 가능 여부는 cluster와 무관하다. Discovery는 저장 Profile과
        # 전일 분위 reference가 중심 종목마다 달라질 수 있어 unit 안에서 고른다.
        _, schedule_reason = _schedule(labels, request, regime)
        discovery_dates = _discovery_dates(labels, request, regime)
        for cluster in request.clusters:
            ordered = universe.members(clusters=(cluster,))[cluster]
            for count in request.per_cluster:
                item: dict[str, Any] = {
                    "regime": regime, "cluster": cluster, "symbol_count": int(count),
                    "state": "SCHEDULE_UNAVAILABLE" if schedule_reason else "DISCOVERY_UNAVAILABLE",
                    "schedule": None,
                    "reason": schedule_reason,
                    "experiment_scope": "DIAGNOSTIC_EOD_REGIME_STRATIFICATION",
                    "eligible_for_algorithm_count": False,
                }
                if schedule_reason is None:
                    attempts = []
                    for discovery_date in discovery_dates:
                        picked = [symbol for symbol in ordered
                                  if (cluster, symbol, discovery_date) in available][:count]
                        if len(picked) != count:
                            attempts.append(f"{discovery_date}: Profile {len(picked)}/{count}")
                            continue
                        reference_date, reference_problem = _search_reference(picked, discovery_date)
                        if reference_problem is not None:
                            attempts.append(f"{discovery_date}: {reference_problem}")
                            continue
                        schedule, reason = _schedule(labels, request, regime, discovery_date)
                        if schedule is None:
                            attempts.append(f"{discovery_date}: {reason}")
                            continue
                        item.update({
                            "state": "READY", "schedule": schedule.as_input(), "reason": None,
                            "discovery_date": discovery_date, "search_reference_date": reference_date,
                            "eligible_core_symbols": picked,
                            "eligible_core_ranks": [ordered.index(symbol) + 1 for symbol in picked],
                        })
                        break
                    if item["state"] != "READY":
                        item["reason"] = "; ".join(attempts) or "해당 시장 상태 Discovery 날짜가 없다"
                        item["state"] = ("DISCOVERY_PROFILE_UNAVAILABLE"
                                         if attempts and all("Profile" in attempt for attempt in attempts)
                                         else "DISCOVERY_SEARCH_REFERENCE_UNAVAILABLE")
                units.append(item)
    payload = {
        "schema": "market_regime_experiment_set.v1", "created_at": now_utc(),
        "request": {**asdict(request), "profile_store": str(request.profile_store),
                    "symbol_day_labels": str(request.symbol_day_labels),
                    "discovery_profit_root": (str(request.discovery_profit_root)
                                               if request.discovery_profit_root is not None else None)},
        "market_label_method": "cross_sectional_median_return_bps_threshold_20",
        "experiment_scope": "DIAGNOSTIC_EOD_REGIME_STRATIFICATION",
        "eligible_for_algorithm_count": False,
        "scope_note": ("당일 종가 수익률 레이블로 날짜를 사후 층화한 분석이다. "
                       "실시간 진입 조건이나 수익 알고리즘 수에 쓰지 않는다."),
        "units": units,
    }
    write_json(output / "experiment_set.json", payload)
    write_json(output / "run_config.json", payload)
    return payload


def run(request: MarketExperimentRequest, output: Path, *, runner: Runner | None = None,
        root: Path = TICK_ROOT) -> dict[str, Any]:
    """준비된 unit만 실행한다. 불가능한 시장 상태·Profile도 결과로 남긴다."""
    output = Path(output)
    frozen = plan(request, output)
    workers = int(request.max_workers)
    if workers < 1:
        raise ValueError("max_workers는 1 이상이어야 한다")
    records = [dict(item) for item in frozen["units"]]
    existing = _resume_records(output) if request.resume else {}
    recovered = _recovered_records(frozen["units"], output,
                                   direct_evidence_only=request.direct_evidence_only)

    def execute(item: dict[str, Any]) -> dict[str, Any]:
        record = dict(item)
        record.setdefault("experiment_scope", "DIAGNOSTIC_EOD_REGIME_STRATIFICATION")
        record.setdefault("eligible_for_algorithm_count", False)
        name = f"{item['regime'].lower()}_{item['cluster'].lower()}_{item['symbol_count']}symbols"
        run_output = output / "runs" / name
        try:
            profile = ProfileSelection(
                clusters=(item["cluster"],), symbols=tuple(item["eligible_core_symbols"]),
                dates=(item["discovery_date"],), store=request.profile_store,
                discovery_profit_root=request.discovery_profit_root)
            schedule = ResearchSchedule(**item["schedule"])
            research = (run_direct_evidence_research(
                DirectEvidenceResearchRequest(profile=profile, schedule=schedule),
                run_output, root=root)
                if request.direct_evidence_only else run_research(
                    ResearchRequest(profile=profile, schedule=schedule), run_output,
                    runner=runner, root=root))
            record.update({"state": research["state"], "output": str(run_output),
                           "artifacts": research["artifacts"],
                           "workflow": ("DIRECT_EVIDENCE_ONLY" if request.direct_evidence_only
                                        else "AGENT_WITH_DIRECT_EVIDENCE_FALLBACK")})
        except Exception as error:
            record.update({"state": "ERROR", "output": str(run_output),
                           "reason": f"{type(error).__name__}: {error}"})
        return record

    ready: list[tuple[int, dict[str, Any]]] = []
    for index, item in enumerate(frozen["units"]):
        if item["state"] != "READY":
            continue
        previous = existing.get(_unit_key(item))
        if (previous is None or previous.get("state") == "READY"):
            previous = recovered.get(_unit_key(item), previous)
        if previous is not None and _same_unit(previous, item):
            if previous.get("state") != "READY" and not (
                    previous.get("state") == "ERROR" and request.retry_errors):
                records[index] = previous
                continue
        ready.append((index, item))

    write_json(output / "experiment_results.json", {
        "schema": "market_regime_experiment_results.v1", "created_at": now_utc(),
        "records": records})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute, item): index for index, item in ready}
        for future in as_completed(futures):
            records[futures[future]] = future.result()
            write_json(output / "experiment_results.json", {
                "schema": "market_regime_experiment_results.v1", "created_at": now_utc(),
                "records": records})
    result = {"output": str(output), "experiment_set": frozen, "records": records}
    write_json(output / "experiment_results.json", {
        "schema": "market_regime_experiment_results.v1", "created_at": now_utc(),
        "records": records})
    return result


def _unit_key(item: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(item.get("regime")), str(item.get("cluster")), int(item.get("symbol_count") or 0))


def _same_unit(previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """같은 key라도 일정·종목이 바뀌었으면 과거 결과를 재사용하지 않는다."""
    return (_same_schedule(previous.get("schedule"), current.get("schedule"))
            and previous.get("discovery_date") == current.get("discovery_date")
            and _stable_json(previous.get("eligible_core_symbols"))
            == _stable_json(current.get("eligible_core_symbols")))


def _same_schedule(left: Any, right: Any) -> bool:
    return (_stable_json(_schedule_for_resume(left))
            == _stable_json(_schedule_for_resume(right)))


def _schedule_for_resume(value: Any) -> Any:
    """새 선택지가 추가되기 전 artifact도 같은 기본값이면 재사용한다."""
    if not isinstance(value, Mapping):
        return value
    normalized = dict(value)
    normalized.setdefault("single_day_discovery_search", False)
    normalized.setdefault("allow_discovery_only_lock", False)
    return normalized


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=list)


def _resume_records(output: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    path = Path(output) / "experiment_results.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    records = payload.get("records") or []
    return {_unit_key(item): dict(item) for item in records if isinstance(item, Mapping)}


def _recovered_records(items: Sequence[Mapping[str, Any]], output: Path, *,
                       direct_evidence_only: bool) -> dict[tuple[str, str, int], dict[str, Any]]:
    """중단된 상위 요약보다 단위별 완료 artifact를 우선해 재개한다."""
    recovered: dict[tuple[str, str, int], dict[str, Any]] = {}
    for item in items:
        if item.get("state") != "READY":
            continue
        name = f"{item['regime'].lower()}_{item['cluster'].lower()}_{item['symbol_count']}symbols"
        path = Path(output) / "runs" / name / "research_run.json"
        if not path.is_file():
            continue
        try:
            saved = read_artifact(path, kind="research_run")["payload"]
        except (OSError, ValueError, KeyError):
            continue
        request = saved.get("request") or {}
        profile = request.get("profile") or {}
        if (tuple(profile.get("clusters") or ()) != (str(item["cluster"]),)
                or tuple(profile.get("symbols") or ())
                != tuple(str(symbol) for symbol in item["eligible_core_symbols"])
                or tuple(profile.get("dates") or ()) != (str(item["discovery_date"]),)
                or not _same_schedule(request.get("schedule"), item.get("schedule"))):
            continue
        state = saved.get("state")
        if not state or state == "READY":
            continue
        recovered[_unit_key(item)] = {
            **dict(item), "state": state, "output": str(path.parent),
            "artifacts": dict(saved.get("artifacts") or {}),
            "workflow": ("DIRECT_EVIDENCE_ONLY" if direct_evidence_only
                         else "AGENT_WITH_DIRECT_EVIDENCE_FALLBACK"),
        }
    return recovered
