"""E0 게이트 — 알려진 미시구조 법칙 다섯 개를 증류로 재발견할 수 있는가.

연구계획서 §4 E0 · ROADMAP.md 3단계. **진입식을 만들지 않는다.** 스칼라 함수
복원만 본다 — 그래서 `sd/compile/`(컴파일러)와 `sd/replay.py`(백테스트)를
전혀 타지 않는다. 단조 흡수(§3 S3)도 적용하지 않는다. 이 패키지가 만드는
것은 스칼라 회귀 하나와 그 복원 판정뿐이다.

다른 모듈과의 관계: 틱 적재(`sd.ticks`)·무차원화(`sd.dimensionless`)·파생
열(`sd.derived`)·on-manifold 표집(`sd.manifold`)·SR 백엔드(`sd.sr`)를 그대로
재사용한다. 여기서 새로 짜는 것은 (1) 다섯 법칙 각각의 입력·타깃 조립
(`targets.py` — 기존 모듈이 계산한 값을 가져다 인덱스 이동·나눗셈만 한다),
(2) 단일-출력 과잉용량 교사(`teacher.py` — `teacher.shallow.ShallowMLP` 의
두 헤드 구조가 E0 의 단일 스칼라 타깃과 맞지 않아 새로 둔다), (3) 복원 판정
함수(`criteria.py`), (4) 세 ablation 조합(`ablation.py`), (5) 오케스트레이션
(`runner.py`) 이다.
"""

from __future__ import annotations

LAWS: tuple[str, ...] = ("L1", "L2", "L3", "L4", "L5")

LAW_TITLES: dict[str, str] = {
    "L1": "마이크로프라이스 (원형 그대로)",
    "L2": "OFI 선형 법칙 (원형 그대로)",
    "L3": "집계 임팩트 제곱근 (프록시로 격하)",
    "L4": "다중 스케일 RV 결합 (대체 형태)",
    "L5": "Hawkes 지수/멱함수 커널 (원형 그대로)",
}

# 거부권이 있는 법칙 — 계획서 §4 게이트 규칙: "L1·L2 동시 실패 → 경고."
VETO_LAWS: tuple[str, ...] = ("L1", "L2")
