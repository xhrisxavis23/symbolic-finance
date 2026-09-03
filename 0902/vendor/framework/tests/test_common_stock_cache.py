from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from framework import featureprofile
from framework.config import read_json, write_json
from framework.contracts.artifacts import create_artifact, write_artifact
from framework.modules import common_stock_cache, evidence_cache
from framework.modules.profile import ProfileSelection


def _listing(path: Path) -> None:
    cells = ["테스트", "KOSPI", "000001", "", "", "2020-01-01", "", "", "", ""]
    row = "".join(f"<td>{value}</td>" for value in cells)
    path.write_text(f"<table><tr>{row}</tr></table>", encoding="euc-kr")


def _selection(store: Path, dates: tuple[str, ...] = ("20260316",)) -> ProfileSelection:
    return ProfileSelection(clusters=(common_stock_cache.SCOPE,), dates=dates,
                            store=store, profile_kind="execution_aligned")


def test_evidence_cache_reuses_same_selection_when_date_order_differs(tmp_path, monkeypatch):
    store = tmp_path / "profile"
    write_json(store / "manifest.json", {"schema": "feature_profile_store.v1", "entries": []})
    first = _selection(store, ("20260317", "20260316"))

    def fake_stage(selection, output):
        artifact = create_artifact("evidence_package", {
            "selection": selection.as_input(), "package": {}})
        write_artifact(Path(output) / "evidence_artifact.json", artifact)
        return artifact

    monkeypatch.setattr(evidence_cache.evidence_stage, "run", fake_stage)
    created = evidence_cache.materialize(first)
    assert created["cache_state"] == "CREATED"
    assert evidence_cache.find(_selection(store, ("20260316", "20260317"))) == Path(created["path"])

    write_json(store / "manifest.json", {"schema": "feature_profile_store.v1", "entries": ["changed"]})
    assert evidence_cache.find(first) is None


def test_common_stock_cache_uses_listing_codes_and_prepares_each_date(tmp_path, monkeypatch):
    listing = tmp_path / "listing.xls"
    _listing(listing)
    tick_root = tmp_path / "ticks"
    (tick_root / "20260316").mkdir(parents=True)
    (tick_root / "20260316" / "000001_quotes.parquet").write_bytes(b"raw")
    (tick_root / "20260316" / "111111_quotes.parquet").write_bytes(b"not-common")
    calls: list[tuple[str, str]] = []

    def fake_data(symbol, date, *, root):
        calls.append((symbol, date))
        return {"cache_state": "CREATED"}

    profile_calls = []

    def fake_profile(scope, symbols, dates, output, *, root):
        profile_calls.append((scope, tuple(symbols), tuple(dates), Path(output)))
        profile = featureprofile.folder(scope, "000001", "20260316", Path(output)) / "profile.parquet"
        profile.parent.mkdir(parents=True, exist_ok=True)
        profile.write_bytes(b"profile")
        write_json(Path(output) / "manifest.json", {"schema": "feature_profile_store.v1"})
        return {"written": len(symbols)}

    evidence_calls = []

    def fake_evidence(selection):
        evidence_calls.append(selection)
        return {"cache_state": "CREATED", "path": "evidence.json", "artifact_id": "E1"}

    monkeypatch.setattr(common_stock_cache.data, "materialize_cache", fake_data)
    monkeypatch.setattr(common_stock_cache.execution_profile, "materialize", fake_profile)
    monkeypatch.setattr(common_stock_cache.evidence_cache, "materialize", fake_evidence)

    store = tmp_path / "profiles"
    result = common_stock_cache.materialize(listing, ("20260316",), profile_store=store,
                                            root=tick_root)

    assert calls == [("000001", "20260316")]
    assert profile_calls == [(common_stock_cache.SCOPE, ("000001",), ("20260316",), store)]
    assert len(evidence_calls) == 1 and evidence_calls[0].clusters == (common_stock_cache.SCOPE,)
    assert result["dates"][0]["excluded_non_common_stock_symbols"] == 1
    manifest = Path(result["manifest"])
    assert manifest.is_file()
    assert "000001" in manifest.read_text(encoding="utf-8")

    common_stock_cache.materialize(listing, ("20260316",), profile_store=store, root=tick_root)
    assert len(profile_calls) == 1


def test_common_stock_cache_accepts_a_dated_listing_directory(tmp_path):
    listings = tmp_path / "listings"
    listings.mkdir()
    _listing(listings / "20260316.xls")
    assert common_stock_cache._listing_for_date(listings, "20260316") == listings / "20260316.xls"


def test_common_stock_cache_uses_date_stock_batch_and_skips_profiles_in_raw_only_mode(
        tmp_path, monkeypatch):
    tick_root = tmp_path / "ticks"
    date_root = tick_root / "20260316"
    date_root.mkdir(parents=True)
    (date_root / "000001_quotes.parquet").write_bytes(b"raw")
    (date_root / "111111_quotes.parquet").write_bytes(b"not-stock")
    pq.write_table(pa.table({
        "issue_code": ["000001", "111111", "0000A0"],
        "asset_type": ["ST", "EF", "ST"],
    }), date_root / "stock_batch_20260316.parquet")
    calls: list[tuple[str, str]] = []

    def fake_data(symbol, date, *, root):
        calls.append((symbol, date))
        return {"cache_state": "CREATED", "source": {"path": symbol}}

    monkeypatch.setattr(common_stock_cache.data, "materialize_cache", fake_data)
    store = tmp_path / "profiles"
    result = common_stock_cache.materialize(
        None, ("20260316",), profile_store=store, root=tick_root,
        stock_batch=True, raw_only=True)

    assert result["state"] == "RAW_STOCK_CACHE_MATERIALIZED"
    assert result["scope"] == common_stock_cache.STOCK_BATCH_SCOPE
    assert calls == [("000001", "20260316")]
    assert result["dates"][0]["selected_stock_count"] == 1
    assert result["dates"][0]["excluded_non_stock_symbols"] == 1
    assert result["dates"][0]["profile"]["state"] == "SKIPPED_RAW_ONLY"


def test_common_stock_profile_cache_checks_metadata_without_loading_arrays(tmp_path, monkeypatch):
    tick_root = tmp_path / "ticks"
    date_root = tick_root / "20260316"
    date_root.mkdir(parents=True)
    (date_root / "000001_quotes.parquet").write_bytes(b"raw")
    pq.write_table(pa.table({"issue_code": ["000001"], "asset_type": ["ST"]}),
                   date_root / "stock_batch_20260316.parquet")
    monkeypatch.setattr(common_stock_cache.data, "cached_source",
                        lambda *_args, **_kwargs: {"path": "cached", "bytes": 1, "mtime_ns": 1})
    monkeypatch.setattr(common_stock_cache.data, "materialize_cache",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("raw array reload")))

    def fake_profile(scope, symbols, dates, output, *, root):
        out = featureprofile.folder(scope, symbols[0], dates[0], output)
        out.mkdir(parents=True)
        (out / "profile.parquet").write_bytes(b"profile")
        write_json(out / "meta.json", {"date": dates[0], "cluster_id": scope,
                                         "symbol": symbols[0], "anchors": 1})
        write_json(Path(output) / "manifest.json", {"schema": "feature_profile_store.v1"})
        return {"written": 1, "skipped": []}

    monkeypatch.setattr(common_stock_cache.execution_profile, "materialize", fake_profile)
    monkeypatch.setattr(common_stock_cache.evidence_cache, "materialize",
                        lambda _selection: {"cache_state": "CREATED"})
    result = common_stock_cache.materialize(None, ("20260316",), profile_store=tmp_path / "profiles",
                                            root=tick_root, stock_batch=True)
    assert result["dates"][0]["raw_cache"]["REUSED"] == 1


def test_profile_build_keeps_completed_raw_cache_available_for_backtest(tmp_path, monkeypatch):
    tick_root = tmp_path / "ticks"
    date_root = tick_root / "20260316"
    date_root.mkdir(parents=True)
    (date_root / "000001_quotes.parquet").write_bytes(b"raw")
    pq.write_table(pa.table({"issue_code": ["000001"], "asset_type": ["ST"]}),
                   date_root / "stock_batch_20260316.parquet")
    store = tmp_path / "profiles"
    manifest = store / common_stock_cache.DIRECTORY / common_stock_cache.STOCK_BATCH_SCOPE / "universe_manifest.json"
    write_json(manifest, {
        "schema": common_stock_cache.SCHEMA,
        "scope": common_stock_cache.STOCK_BATCH_SCOPE,
        "selection_definition": "날짜별 stock_batch의 asset_type=ST 이면서 6자리 issue_code (ETF/ETN 제외)",
        "selection_input": str(tick_root.resolve()),
        "state": "COMPLETE",
        "dates": {"20260316": {"symbols": ["000001"], "raw_cache": {"REUSED": 1}}},
    })
    monkeypatch.setattr(common_stock_cache.data, "cached_source",
                        lambda *_args, **_kwargs: {"path": "cached", "bytes": 1, "mtime_ns": 1})

    def fake_profile(scope, symbols, dates, output, *, root):
        out = featureprofile.folder(scope, symbols[0], dates[0], output)
        out.mkdir(parents=True)
        (out / "profile.parquet").write_bytes(b"profile")
        write_json(out / "meta.json", {"date": dates[0], "cluster_id": scope,
                                         "symbol": symbols[0], "anchors": 1})
        write_json(Path(output) / "manifest.json", {"schema": "feature_profile_store.v1"})
        return {"written": 1, "skipped": []}

    monkeypatch.setattr(common_stock_cache.execution_profile, "materialize", fake_profile)
    monkeypatch.setattr(common_stock_cache.evidence_cache, "materialize",
                        lambda _selection: {"cache_state": "CREATED"})
    common_stock_cache.materialize(None, ("20260316",), profile_store=store,
                                   root=tick_root, stock_batch=True)
    stored = read_json(manifest)
    assert stored["state"] == "COMPLETE"
    assert stored["profile_state"] == "COMPLETE"
