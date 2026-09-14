"""교사 점수 주입 — PREREG-TEACHER-ECON.md §3.3. **vendor 파일은 한 글자도 바꾸지 않는다.**

이 모듈을 `install()` 한 프로세스(와 그 뒤에 fork 된 재생 워커)에서만 두 함수를 감싼다.

1. `framework.data.load` — 워커가 지금 읽는 (종목, 날짜)를 기록한다.
2. `framework.contract.ExpressionRuntime.evaluate` — 표식 AST 를 만나면 그 종목-일의 교사 점수를 돌려준다.
   진입 신호의 입력식과 분위 임계의 입력식이 둘 다 이 함수를 거치므로 둘 다 교사 점수로 계산된다.

C2(틱 정렬): 점수 파일의 `time_s`·BID1 이 실행기가 가진 배열과 다르면 오류로 멈춘다.
"""
from __future__ import annotations

import numpy as np

from . import common as C

from framework import contract as fcontract  # noqa: E402  (common 이 vendor 경로를 잡았다)
from framework import data as fdata  # noqa: E402

MARKER_KEYS = {fcontract.sha256_json(ast): model for model, ast in C.MARKERS.items()}
_CURRENT: dict = {}
_CACHE: dict = {}
_INSTALLED = False


def _score_for(model: str, runtime) -> np.ndarray:
    symbol, date = _CURRENT.get("symbol"), _CURRENT.get("date")
    if symbol is None:
        raise RuntimeError("C2 실패: 현재 종목-일을 모른다 (data.load 를 거치지 않은 배열)")
    if date != C.DATE:
        raise RuntimeError(f"C2 실패: 날짜가 {C.DATE} 가 아니다: {date}")
    key = (model, symbol)
    if key not in _CACHE:
        path = C.score_path(model, symbol)
        if not path.exists():
            raise KeyError(f"교사 점수 파일이 없다: {model} {symbol}")
        if len(_CACHE) > 16:
            _CACHE.clear()
        z = np.load(path)
        _CACHE[key] = (z["time_s"], z["bid1"], z["score"])
    time_s, bid1, score = _CACHE[key]
    data = runtime.book.data
    if (runtime.n != len(score)
            or not np.array_equal(np.asarray(data["time_s"], dtype=float), time_s)
            or not np.array_equal(np.asarray(data["bid_price"], dtype=float)[:, 0], bid1, equal_nan=True)):
        raise RuntimeError(f"C2 실패: {model} {symbol} 점수 파일과 실행기 배열의 틱이 다르다 "
                           f"(n {runtime.n} vs {len(score)})")
    return score


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    original_load = fdata.load

    def load(*args, **kwargs):
        arrays, path = original_load(*args, **kwargs)
        symbol = args[0] if args else kwargs["symbol"]
        date = args[1] if len(args) > 1 else kwargs["date"]
        _CURRENT.clear()
        _CURRENT.update(symbol=str(symbol), date=str(date))
        return arrays, path

    original_evaluate = fcontract.ExpressionRuntime.evaluate

    def evaluate(self, expression):
        key = fcontract.sha256_json(expression)
        model = MARKER_KEYS.get(key)
        if model is None:
            return original_evaluate(self, expression)
        if key not in self._cache:
            self._cache[key] = _score_for(model, self)
        return self._cache[key]

    fdata.load = load
    fcontract.ExpressionRuntime.evaluate = evaluate
    _INSTALLED = True
