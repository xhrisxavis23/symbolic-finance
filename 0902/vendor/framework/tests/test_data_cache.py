from __future__ import annotations

import numpy as np

from framework import data


def _arrays(value: float = 1.0) -> dict[str, np.ndarray]:
    return {
        "bid_price": np.full((3, 10), value),
        "ask_price": np.full((3, 10), value + 1.0),
        "bid_qty": np.full((3, 10), 2.0),
        "ask_qty": np.full((3, 10), 3.0),
        "local_time": np.array([90000000000, 90001000000, 90002000000]),
        "time_s": np.array([0.0, 1.0, 2.0]),
        "buy_volume": np.array([0.0, 2.0, 0.0]),
    }


def test_backtest_cache_is_used_only_when_its_source_matches(tmp_path, monkeypatch):
    source = tmp_path / "001_20260407.parquet"
    source.write_bytes(b"raw-v1")
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(data, "find_parquet", lambda symbol, date, root: source)
    monkeypatch.setattr(data, "_load_parquet", lambda path, symbol, date: (_arrays(), path))

    created = data.materialize_cache("001", "20260407", root=tmp_path,
                                     cache_root=cache_root)
    assert created["cache_state"] == "CREATED"

    def cache_must_be_used(path, symbol, date):
        raise AssertionError("valid cache를 두고 원본 parquet을 읽었다")

    monkeypatch.setattr(data, "_load_parquet", cache_must_be_used)
    arrays, path = data.load("001", "20260407", tmp_path, cache_root=cache_root)
    assert path == source
    assert arrays["bid_price"].shape == (3, 10)
    assert arrays["buy_volume"].tolist() == [0.0, 2.0, 0.0]

    source.write_bytes(b"raw-v2-is-different")
    replacement = _arrays(9.0)
    monkeypatch.setattr(data, "_load_parquet", lambda path, symbol, date: (replacement, path))
    arrays, _ = data.load("001", "20260407", tmp_path, cache_root=cache_root)
    assert arrays["bid_price"][0, 0] == 9.0


def test_backtest_cache_uses_its_saved_source_path_before_directory_glob(tmp_path, monkeypatch):
    source = tmp_path / "001_20260407.parquet"
    source.write_bytes(b"raw-v1")
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(data, "find_parquet", lambda symbol, date, root: source)
    monkeypatch.setattr(data, "_load_parquet", lambda path, symbol, date: (_arrays(), path))
    data.materialize_cache("001", "20260407", root=tmp_path, cache_root=cache_root)

    monkeypatch.setattr(data, "find_parquet", lambda *_args: (_ for _ in ()).throw(
        AssertionError("유효한 array cache인데 directory glob을 했다")))
    arrays, path = data.load("001", "20260407", tmp_path, cache_root=cache_root)

    assert path == source
    assert arrays["time_s"].tolist() == [0.0, 1.0, 2.0]
