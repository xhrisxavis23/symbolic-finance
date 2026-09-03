"""종목 우주. 어느 클러스터에 어느 종목이 있는가.

구 구조(`analysis/iasdva/`, 2026-09-02 삭제)에서 이 명단은 `agents.py` 의
`CORE_MEMBERS` 상수와 `__main__.py` 두 곳에서 각각 열려 `["groups"]` 를 꺼냈다. 클러스터
설명(무엇이 MK04 를 MK04 로 만드는가)은 코드 안에 아예 없고 포털 JSON 에만 있어서,
Agent 에게 클러스터를 알려줄 방법이 없었다.

여기서는 **명단과 그 뜻이 한 곳**이다. `catalog.py` 가 feature 에 대해 하는 일을
클러스터에 대해 한다.

명단 파일은 클러스터링 run 이 만든 산출물이라 손으로 고치지 않는다. 파일의 sha256
(`1dc3b0cf...`) 이 수익 구간 캐시의 `cache_config.json` 에 기록돼 있어서, 내용이
바뀌면 그 캐시와의 연결이 끊긴다. `members_sha256()` 으로 언제든 대조한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import read_json, sha256_file


MEMBERS_PATH = Path(__file__).with_name("microstructure_cluster_core_members.json")
MEMBERS_SCHEMA = "microstructure_cluster_core_members.v1"

# 캐시가 기록해 둔 명단 해시. 이 값과 어긋나면 캐시와 명단이 다른 세계다.
CACHED_MEMBERS_SHA256 = "1dc3b0cf996a94ccab92b1434813dbcd5292efb4c1af11bf1d8a18c6f3704b20"


@dataclass(frozen=True)
class Cluster:
    """클러스터 한 행. 이름·뜻·전체 규모가 한 자리에 있다."""

    key: str
    label: str
    universe_size: int      # 클러스터링 전체 2,668 종목 중 이 그룹의 크기
    doc: str                # 무엇이 이 그룹을 다른 그룹과 구별하는가
    spread_bps: float       # 중앙값. 지정가 전략에서 수수료 다음으로 큰 마찰
    ask_l1_depth: float     # 매도 1호가 잔량. 지정가 매수 큐가 얼마나 긴가
    rv_20t_bps: float       # 20틱 실현변동성. 30초 안에 움직일 여지


# 출처: 2,668 종목 x 20260316~20260424, 6축 k-means, silhouette 0.220, PCA 71.5%.
# 미래 수익·PnL·체결 라벨을 쓰지 않고 나눴다 — 그래서 클러스터로 자르는 것 자체는
# look-ahead 가 아니다.
CLUSTERS: dict[str, Cluster] = {
    c.key: c for c in (
        Cluster("MK01", "고활동·고유동성형", 411,
                "체결·호가 이벤트가 많고 주문 처리가 쉽다. 왕복 마찰이 작다",
                spread_bps=9.64, ask_l1_depth=703.0, rv_20t_bps=8.75),
        Cluster("MK02", "매도호가취약·저유동성형", 378,
                "매도 1호가가 얇아 소진에 민감하다. 유동성과 활동성이 낮다",
                spread_bps=16.26, ask_l1_depth=84.25, rv_20t_bps=18.27),
        Cluster("MK03", "초고변동성·초고비용형", 174,
                "짧은 가격 움직임이 매우 크고 왕복 마찰도 매우 크다",
                spread_bps=51.98, ask_l1_depth=158.25, rv_20t_bps=66.38),
        Cluster("MK04", "저호가이동·보통활동형", 846,
                "호가 가격이 잘 움직이지 않는다. 활동성·유동성은 보통보다 조금 위",
                spread_bps=17.07, ask_l1_depth=243.5, rv_20t_bps=17.95),
        Cluster("MK05", "초고호가이동·고활동형", 85,
                "호가 가격이 매우 자주 재배치된다. 잔량이 두껍고 활동성이 높다",
                spread_bps=8.88, ask_l1_depth=3576.0, rv_20t_bps=7.25),
        Cluster("MK06", "고비용·고변동성형", 610,
                "왕복 마찰이 크고 짧은 변동성도 크다. 호가가 자주 재배치된다",
                spread_bps=31.46, ask_l1_depth=148.75, rv_20t_bps=34.44),
        Cluster("MK07", "고유동성·저호가이동형", 164,
                "주문 처리가 쉽고 호가 가격이 잘 움직이지 않는다",
                spread_bps=12.43, ask_l1_depth=1496.5, rv_20t_bps=8.33),
    )
}


def _groups() -> dict[str, list[str]]:
    payload = read_json(MEMBERS_PATH)
    if payload.get("schema") != MEMBERS_SCHEMA:
        raise ValueError(f"명단 스키마가 다르다: {payload.get('schema')}")
    return {str(key): [str(s) for s in value] for key, value in payload["groups"].items()}


def members(
    per_cluster: int | None = None,
    *,
    clusters: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    """클러스터마다 종목 명단. `campaign.run(members=...)` 에 그대로 넣는다.

    `per_cluster` 는 앞에서부터 자른다. 무작위로 뽑지 않는다 — 뽑는 규칙이 결과를
    좌우하면 그 run 을 다시 만들 수 없다. 파일의 순서가 곧 정본이다.
    """
    groups = _groups()
    unknown = set(clusters or ()) - set(groups)
    if unknown:
        raise KeyError(f"명단에 없는 클러스터: {sorted(unknown)}")
    return {
        key: (list(value) if per_cluster is None else list(value)[: int(per_cluster)])
        for key, value in groups.items()
        if not clusters or key in clusters
    }


def symbols(
    per_cluster: int | None = None,
    *,
    clusters: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """평평한 종목 목록. `loop.run(symbols=...)` 이 받는 형태."""
    flat: list[str] = []
    for value in members(per_cluster, clusters=clusters).values():
        flat.extend(value)
    return tuple(flat)


def cluster_of(symbol: str) -> str | None:
    """이 종목이 속한 클러스터. 명단에 없으면 None."""
    for key, value in _groups().items():
        if str(symbol) in value:
            return key
    return None


def members_sha256() -> str:
    return sha256_file(MEMBERS_PATH)


def agent_vocabulary(clusters: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Agent 에게 줄 클러스터 설명.

    클러스터 이름만 주면 Agent 는 `MK04` 가 무엇인지 모른 채 가설을 쓴다. 스프레드와
    잔량을 함께 줘야 "지정가로 살 수 있는가" 를 가설 안에서 따질 수 있다.
    """
    counts = {key: len(value) for key, value in _groups().items()}
    return [
        {"cluster_id": c.key, "label": c.label, "doc": c.doc,
         "core_symbols": counts.get(c.key, 0), "universe_size": c.universe_size,
         "median_spread_bps": c.spread_bps, "median_ask_l1_depth": c.ask_l1_depth,
         "median_rv_20t_bps": c.rv_20t_bps}
        for c in sorted(CLUSTERS.values(), key=lambda x: x.key)
        if clusters is None or c.key in set(clusters)
    ]


def audit() -> dict[str, Any]:
    """명단과 설명이 어긋나지 않는지. 기동 시 한 번 부른다."""
    groups = _groups()
    digest = members_sha256()
    return {
        "clusters": sorted(groups),
        "core_symbols": {key: len(value) for key, value in sorted(groups.items())},
        "total_core_symbols": sum(len(value) for value in groups.values()),
        "described_only": sorted(set(CLUSTERS) - set(groups)),
        "listed_only": sorted(set(groups) - set(CLUSTERS)),
        "duplicate_symbols": sorted(
            symbol for symbol in {s for value in groups.values() for s in value}
            if sum(value.count(symbol) for value in groups.values()) > 1
        ),
        "members_sha256": digest,
        "matches_cache": digest == CACHED_MEMBERS_SHA256,
    }
