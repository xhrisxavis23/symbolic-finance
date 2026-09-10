# RESULTS — E1 본 실험 실행 결과

`PREREG.md` 에 잠근 파라미터로 실행했다. 시드 [0, 1, 2], 분위격자 [0.7, 0.85, 0.95], niterations=200, maxsize=18, max_samples=200,000.

## 최종 판정

**성공 주장 불가 — 3개 시드 중 전부가 통과하지 못했다 (R37-1)**

| 시드 | 게이트 통과 | H total_net_bps>0 | H scorable≥200 | H 에서 영점(h) 대비 우위 | 종합 판정 |
| --- | --- | --- | --- | --- | --- |
| seed0 | False | False | True | False | 실패 |
| seed1 | False | False | True | True | 실패 |
| seed2 | False | False | True | True | 실패 |

## L·M·H 저하 곡선 (R37-2)

L 은 학습에 쓴 층이라 성과 근거가 아니라 저하 곡선의 기준점이다. M 은 선택에, H 는 최종 평가에 썼다(봉인 층).

### seed0

| 층 | total_net_bps | bps_per_decision | scorable |
| --- | --- | --- | --- |
| L(학습, 기준점) | -279574.78 | -2.4442 | 114382 |
| M(선택) | -159976.13 | -1.8286 | 87486 |
| H(최종, 봉인) | -93372.30 | -1.7718 | 52700 |

### seed1

| 층 | total_net_bps | bps_per_decision | scorable |
| --- | --- | --- | --- |
| L(학습, 기준점) | -120324.64 | -0.9797 | 122818 |
| M(선택) | -47799.75 | -0.5755 | 83054 |
| H(최종, 봉인) | -19300.81 | -0.4153 | 46475 |

### seed2

| 층 | total_net_bps | bps_per_decision | scorable |
| --- | --- | --- | --- |
| L(학습, 기준점) | -125322.02 | -0.9737 | 128704 |
| M(선택) | -45575.18 | -0.5283 | 86267 |
| H(최종, 봉인) | -22015.39 | -0.4597 | 47889 |

### naive

| 층 | total_net_bps | bps_per_decision | scorable |
| --- | --- | --- | --- |
| L(학습, 기준점) | -148566.87 | -1.1929 | 124538 |
| M(선택) | -64130.10 | -0.7808 | 82136 |
| H(최종, 봉인) | -37997.68 | -0.7975 | 47647 |

### null

| 층 | total_net_bps | bps_per_decision | scorable |
| --- | --- | --- | --- |
| L(학습, 기준점) | -239697.99 | -1.7730 | 135193 |
| M(선택) | -121593.02 | -1.2398 | 98076 |
| H(최종, 봉인) | -67537.52 | -1.2223 | 55256 |

## H2② 일중 외삽 — H, 13:00 이후만

| label | total_net_bps | bps_per_decision | scorable |
| --- | --- | --- | --- |
| seed0 | -25506.12 | -1.3310 | 19163 |
| seed1 | -5839.55 | -0.3348 | 17442 |
| seed2 | -5466.14 | -0.3058 | 17875 |
| naive | -9337.58 | -0.5392 | 17319 |
| null | -18852.99 | -0.9618 | 19601 |

## 구조 복원율

전체 SR 후보(전 시드 합) 18개 중 서로 다른 구조(normal_form) 14개.

가장 흔한 구조 (구조, 등장 횟수):

- `Symbol('microprice_velocity_over_spread')` — 3회
- `Add(Symbol('book_imbalance'), Symbol('signed_aggr_flow_100'), Symbol('signed_aggr_flow_20'), Symbol('spread_to_round_trip_cost_ratio'))` — 2회
- `Add(Symbol('book_imbalance'), Symbol('signed_aggr_flow_100'), Symbol('spread_to_round_trip_cost_ratio'), tanh(Symbol('signed_aggr_flow_20')))` — 2회
- `Symbol('signed_aggr_flow_20')` — 1회
- `Add(Symbol('book_imbalance'), Symbol('signed_aggr_flow_20'))` — 1회
- `Add(Symbol('book_imbalance'), Symbol('signed_aggr_flow_20'), Symbol('spread_to_round_trip_cost_ratio'))` — 1회
- `Add(Symbol('book_imbalance'), Symbol('deep_depth_imbalance_6_10'), Symbol('signed_aggr_flow_100'), Symbol('signed_aggr_flow_20'), Symbol('spread_to_round_trip_cost_ratio'))` — 1회
- `Add(Symbol('book_imbalance'), Symbol('signed_aggr_flow_100'), Symbol('spread_to_round_trip_cost_ratio'), tanh(Add(Symbol('deep_depth_imbalance_6_10'), Symbol('signed_aggr_flow_20'))))` — 1회
- `Add(Symbol('book_imbalance'), Symbol('signed_aggr_flow_100'), Symbol('spread_to_round_trip_cost_ratio'), tanh(Add(Symbol('signed_aggr_flow_20'), tanh(Symbol('deep_depth_imbalance_6_10')))))` — 1회
- `Symbol('signed_aggr_flow_100')` — 1회

## 시드별 진단

### seed=0

- 게이트 통과: False
- 게이트 상세: {"beats_constant": {"passed": true, "correlation": 0.13489411967444576, "threshold": 0.05}, "fill_calibration": {"passed": false, "max_abs_gap": 0.3730836016967483, "threshold": 0.25}, "adverse_selection_sign": {"passed": true, "top_decile_mean_y_path": -1.1283257741320492, "top_decile_size": 19806, "why": "양수면 큐 모델 낙관 또는 라벨 누수를 의심한다"}}
- 교사 학습 1461초, 조기종료 stopped_epoch=180 best_epoch=80
- SR 적합 1881초, 후보 14개, 컴파일 성공 1/14
- 1순위 후보: `microprice_velocity_over_spread` (임계 분위 0.95)

### seed=1

- 게이트 통과: False
- 게이트 상세: {"beats_constant": {"passed": true, "correlation": 0.12613493014210003, "threshold": 0.05}, "fill_calibration": {"passed": false, "max_abs_gap": 0.2785777150605921, "threshold": 0.25}, "adverse_selection_sign": {"passed": true, "top_decile_mean_y_path": -1.1224938134233016, "top_decile_size": 19797, "why": "양수면 큐 모델 낙관 또는 라벨 누수를 의심한다"}}
- 교사 학습 1348초, 조기종료 stopped_epoch=160 best_epoch=60
- SR 적합 2039초, 후보 14개, 컴파일 성공 9/14
- 1순위 후보: `tanh((book_imbalance + signed_aggr_flow_20 + spread_to_round_trip_cost_ratio)/16.481377) - 1.063918` (임계 분위 0.95)

### seed=2

- 게이트 통과: False
- 게이트 상세: {"beats_constant": {"passed": true, "correlation": 0.12475070943675394, "threshold": 0.05}, "fill_calibration": {"passed": false, "max_abs_gap": 0.4027317445519151, "threshold": 0.25}, "adverse_selection_sign": {"passed": true, "top_decile_mean_y_path": -1.127452484080525, "top_decile_size": 19798, "why": "양수면 큐 모델 낙관 또는 라벨 누수를 의심한다"}}
- 교사 학습 1278초, 조기종료 stopped_epoch=150 best_epoch=50
- SR 적합 1923초, 후보 14개, 컴파일 성공 8/14
- 1순위 후보: `tanh(book_imbalance + queue_imbalance_best*signed_aggr_flow_20 + signed_aggr_flow_100 + spread_to_round_trip_cost_ratio)*0.13854784 - 1*1.0886018` (임계 분위 0.95)

## 비교군

- (f) 나이브(`book_imbalance`): `book_imbalance`
- (h) 영점 근사(`book_imbalance_velocity`): `book_imbalance_velocity`
- (a)(c)(g) 및 ablation 트랙·병목 ablation은 이번 실행 범위 밖(`run/locked_params.py::NOT_IN_SCOPE` 참조)

## 시도 횟수 N

- SR 적합: 시드 3 × 트랙 1(main) = 3회
- M 선택 재생 attempts: 60 (전 시드 컴파일 후보 합 × 분위격자 3 + 비교군 2 × 분위격자)
- M 선택에서 자격을 얻은 시드: 3/3
- L/M/H 곡선 재생 attempts: 동결 후보 수 × 분위격자 × 2개 층(L,H)

## 봉인 홀드아웃(`20260317`)

**이 실행은 봉인 홀드아웃을 열지 않았다.** DESIGN.md §5.1 의 조건(H 통과 + 3시드 전부 통과)을 만족하는지는 위 표로 판단한다. 만족하면 별도 단계로 연다 — 이 파일이 그 결정을 자동으로 내리지 않는다.

