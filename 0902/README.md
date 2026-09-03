# 0902 — 심볼릭 증류 얇은 수직 슬라이스

`20260316` KRX 주식 30종목으로 데이터 적재부터 정본 백테스트 원장까지
파이프라인 전 단계를 관통시킨다.

**최종 산출은 LONG 진입 조건 하나다. 청산은 탐색하지 않는다** —
`CANONICAL_QUEUE_V9` 가 gross −120bp 손절 · 최고 ASK1 대비 −30bp 추적 ·
최대 900초 · 왕복 23bp 를 상수로 건다.

## 문서

| 문서 | 내용 |
| --- | --- |
| [`../symbolic-distillation-for-market-microstructure.md`](../symbolic-distillation-for-market-microstructure.md) | 계획서 — 무엇을 왜 |
| [`DESIGN.md`](./DESIGN.md) | 설계와 결정 기록 D1~D13 |
| [`PLAN.md`](./PLAN.md) | 구현 계획 |
| [`vendor/PATCHES.md`](./vendor/PATCHES.md) | 정본 프레임워크 사본의 변경 내역 |

## 실행

```bash
make vendor-diff     # vendor 가 원본과 PATCHES.md 외에 다르지 않은지
make test-fast       # 합성 데이터 테스트만 (수 초)
make test            # 실데이터 테스트 포함 (수 분)
make slice           # 엔드투엔드
```

산출물은 `runs/<UTC타임스탬프>-<설정해시>/` 아래에 쌓인다.

| 파일 | 내용 |
| --- | --- |
| `universe.json` | 30종목 명단과 층 경계 |
| `candidates.json` | SR 후보 전체 (선택된 것만이 아니라) |
| `compile_report.json` | 성공 목록 + **실패 목록과 단계별 사유** |
| `entry_expressions/*.json` | Catalog AST 진입식 |
| `ledger.parquet` | 정본 재생 원장 |
| `summary.parquet` | 계약별 성과 |
| `provenance.json` | 프로필·Catalog·vendor 해시, 시드, SR 백엔드, 시도 횟수 N |
| `report.md` | 사람이 읽는 요약 |

## 지금 이 슬라이스가 검증하는 것

**배관이다. 알파가 아니다.** 교사는 얕은 MLP 이고 SR 백엔드는 템플릿 격자다.
성과 숫자는 어떤 가설 판정에도 쓸 수 없다 (DESIGN.md D3·D4).
리포트가 이것을 맨 위에 표시한다.

## 다음

교사를 `DeepLOBCompact` 로 교체 → SR 을 PySR 로 교체 → E0 게이트 5종 →
전종목 캐시 → 본 실험 비교군 8종.
