"""PREREG.md 가 잠근 파라미터의 단일 진실 원천. 모든 스테이지 스크립트가
여기서 값을 가져온다 — 스크립트마다 값을 따로 적으면 결과를 보고 슬쩍
바꿀 자리가 생긴다.

이 파일 자체가 사전등록의 일부다. `PREREG.md` 커밋과 같은 커밋에 들어간다.
**실행 시작 이후 이 파일을 바꾸지 않는다** — 바꿔야 한다면 그 사실과 이유를
PROGRESS.md 에 남기고 별도 커밋으로 한다(조용히 덮어쓰지 않는다).
"""

from __future__ import annotations

DATE = "20260316"
SEEDS: tuple[int, ...] = (0, 1, 2)                 # R37-1 — 3개, 전부 통과해야 주장

# 시간 창 (time_s, 초 단위 — 09:00:00=32400, 12:00:00=43200,
# 13:00:00=46800, 15:30:00=55800). DESIGN.md §2.2.
T_0900 = 32400.0
T_1200 = 43200.0
T_1300 = 46800.0
T_1530 = 55800.0

L_TRAIN_WINDOW = (T_0900, T_1200)          # 교사 학습·SR 적합
M_SELECT_WINDOW = (T_1200, T_1300)         # 조기 종료·후보 선택
FULL_DAY_WINDOW = (T_0900, T_1530)         # L/M/H 저하 곡선(R37-2), 최종 평가
H_INTRADAY_WINDOW = (T_1300, T_1530)       # H2② 일중 외삽 진단(사후, 원장에서 자름)

# 교사 — max_samples 는 SR 과 공유한다(R37-4, D13). 분리하지 않는다.
MAX_SAMPLES = 200_000
BOTTLENECK = 2                              # 병목 ablation(d=1,3,8,32)은 이번 실행 범위 밖(§후속)
TEACHER_WINDOW = 16
TEACHER_EPOCHS = 700
TEACHER_EVAL_EVERY = 10
TEACHER_PATIENCE = 10                       # = 100epoch 무개선

# SR — R37-3: niterations 원안 그대로, 낮추지 않는다.
SR_NITERATIONS = 200
SR_MAXSIZE = 18                              # 0909 G2 선례(계획서 상한 20 이내)
SR_DETERMINISTIC = True                      # 구조 복원율 판정에 필요(직렬)

# 임계 분위 격자 — 결과를 보기 전에 고정한다(0902 D8 선례).
QUANTILE_GRID: tuple[float, ...] = (0.70, 0.85, 0.95)

# S4 선택 규칙 — 분모 함정 방지(계획서 §3 S4)
MIN_SCORABLE = 200

# 비교군 — 이번 실행 범위. (a) 교사 자체는 Catalog AST 실행기가 신경망
# 점수를 직접 평가할 수 없어 이번 범위에서 제외한다(아래 NOT_IN_SCOPE 참조).
# (f) 나이브·(h) 영점은 계획서 §5 통과선이 요구하는 최소 비교군이다. 둘 다
# SR 후보와 **같은 분위 격자(QUANTILE_GRID)** 에서 컷한다 — 절대 리터럴
# 임계(예: "0.5")는 컴파일 파이프라인이 기본으로 지원하지 않고(분위만
# 지원), 컴파일 경로를 SR 후보와 다르게 타면 "같은 조건에서 비교"가
# 깨진다. 그래서 절대 임계 대신 같은 분위 컷 메커니즘을 그대로 쓴다.
NAIVE_FEATURE = "book_imbalance"             # (f) — 방향(불균형)만 보는 가장 단순한 규칙
NULL_FEATURE = "book_imbalance_velocity"     # (h) 근사 — 사유는 PREREG.md "무작위 기준선" 절

NOT_IN_SCOPE = (
    "(a) 교사 자체 비교 — Catalog AST 실행기가 신경망 순전파를 직접 평가할 "
    "방법이 없다. 0902 를 확장해야 하는 새 엔지니어링이라 이번 실행 범위 "
    "밖이다.",
    "(b)(d)(e) ablation 트랙 — no_distillation · no_dimensionless · "
    "uniform_off_manifold. 후속 실행으로 이연.",
    "(c) OFI 기준선 — (f)(h) 만으로 계획서 §5 통과선을 판정할 수 있다. "
    "성공하면 후속으로 추가.",
    "(g) Agent 산 진입식 — 별도의 무거운 워크플로(Optuna 등)라 이번 실행에 "
    "포함하지 않는다.",
    "병목 차원 ablation d∈{1,3,8,32} — d=2(BOTTLENECK) 만 이번 실행에서 돈다.",
)
