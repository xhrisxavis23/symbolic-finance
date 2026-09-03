"""Search, Validation Backtest, Final Backtest의 날짜 블록을 Workflow 입력으로 명시한다."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ResearchSchedule:
    validation_dates: tuple[str, ...]
    search_fit_dates: tuple[str, ...]
    search_confirm_dates: tuple[str, ...]
    backtest_dates: tuple[str, ...]
    terminal_oos_dates: tuple[str, ...]
    single_day_discovery_search: bool = False
    allow_discovery_only_lock: bool = False
    # 기존 workflow 입력 이름. 새 이름과 같은 뜻으로 계속 받는다.
    search_on_discovery_dates: bool = False

    def as_input(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def discovery_only_lock(self) -> bool:
        """별도 Search confirm 없이 Discovery fit 결과를 Validation에 넘길 수 있는가."""
        return bool(self.single_day_discovery_search or self.allow_discovery_only_lock
                    or self.search_on_discovery_dates)

    def validate(self, discovery_dates: tuple[str, ...]) -> None:
        """각 선택·검증·최종 실행은 서로 다른 날짜만 쓴다."""
        named = {
            "discovery": set(discovery_dates),
            "validation": set(self.validation_dates),
            "search_fit": set(self.search_fit_dates),
            "search_confirm": set(self.search_confirm_dates),
            "backtest": set(self.backtest_dates),
            "terminal_oos": set(self.terminal_oos_dates),
        }
        for name in ("validation", "backtest"):
            if not named[name]:
                raise ValueError(f"{name} 날짜가 비었다")
        if not named["search_fit"]:
            raise ValueError("search_fit 날짜가 비었다")
        if self.discovery_only_lock:
            if named["search_fit"] != named["discovery"]:
                raise ValueError("discovery-only lock의 Search fit은 Discovery 날짜와 같아야 한다")
            if named["search_confirm"]:
                raise ValueError("discovery-only lock에는 search_confirm 날짜를 넣을 수 없다")
            blocks = ("discovery", "validation", "backtest")
        else:
            blocks = ("discovery", "search_fit", "search_confirm", "validation", "backtest")
        for index, left in enumerate(blocks):
            for right in blocks[index + 1:]:
                shared = sorted(named[left] & named[right])
                if shared:
                    raise ValueError(f"{left}와 {right} 날짜가 겹친다: {shared}")
        consumed = set().union(*(named[name] for name in blocks))
        leaked = sorted(consumed & named["terminal_oos"])
        if leaked:
            raise ValueError(f"terminal_oos 날짜를 앞 단계가 열려고 한다: {leaked}")
