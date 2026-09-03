from __future__ import annotations

from pathlib import Path

from framework import executable
from framework.config import write_json
from framework.specification_replay import _replay_specification, _source_records


def test_specification_replay_migrates_prior_day_q_and_deduplicates_same_entry(tmp_path):
    root = tmp_path / "runs"
    for cluster in ("A_UP", "B_ALL"):
        path = root / cluster / "03_implementation" / "units" / "H" / "source_specification.json"
        write_json(path, {
            "hypothesis_id": "H",
            "semantic_invariants": {"hypothesis_id": "H", "direction": "HIGHER"},
            "signal_template": {
                "op": "compare", "input": {"op": "primitive", "primitive_id": "book_imbalance"},
                "comparator": ">", "value": "UNRESOLVED:q_book_imbalance",
                "_parameter": {"threshold_source": {
                    "kind": executable.THRESHOLD_PRIOR_VALID_DAY_SYMBOL_QUANTILE}},
            },
            "search_boundary": {"allowed_parameters": [{
                "name": "q_book_imbalance", "feature": "book_imbalance", "direction": "HIGHER",
                "threshold_source": {"kind": executable.THRESHOLD_PRIOR_VALID_DAY_SYMBOL_QUANTILE},
            }], "fixed_semantics": {"canonical_feature": "book_imbalance"}},
        })

    records = _source_records(root)
    assert len(records) == 2
    assert len({record["signature"] for record in records}) == 1
    replay_id, spec = _replay_specification(records[0])
    assert replay_id.startswith("Q100_")
    assert spec["hypothesis_id"] == spec["semantic_invariants"]["hypothesis_id"]
    assert (spec["signal_template"]["_parameter"]["threshold_source"]["kind"]
            == executable.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE)
    assert (spec["search_boundary"]["allowed_parameters"][0]["threshold_source"]["kind"]
            == executable.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE)


def test_specification_replay_source_manifest_uses_only_listed_composites(tmp_path):
    root = tmp_path / "runs"
    selected = root / "A" / "03_implementation" / "units" / "SELECTED" / "source_specification.json"
    excluded = root / "B" / "03_implementation" / "units" / "EXCLUDED" / "source_specification.json"
    for path, hypothesis_id in ((selected, "SELECTED"), (excluded, "EXCLUDED")):
        write_json(path, {
            "hypothesis_id": hypothesis_id,
            "signal_template": {"op": "compare",
                                "input": {"op": "primitive", "primitive_id": "book_imbalance"},
                                "comparator": ">", "value": "UNRESOLVED:q"},
            "search_boundary": {"allowed_parameters": [{
                "name": "q", "feature": "book_imbalance", "direction": "HIGHER",
                "threshold_source": {"kind": executable.THRESHOLD_PRIOR_VALID_DAY_SYMBOL_QUANTILE},
            }]},
        })
    manifest = tmp_path / "composites.json"
    write_json(manifest, {"candidates": [{"source_specification_path": str(selected)}]})

    records = _source_records(root, manifest)

    assert [record["source_hypothesis_id"] for record in records] == ["SELECTED"]
