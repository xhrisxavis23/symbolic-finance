from framework import mechanism_graph as M


def graph(*, mode="HOLD_THROUGH", pending_ids=None, bindings=(), context_at="DECISION"):
    return {
        "schema": M.SCHEMA_VERSION,
        "observables": [
            {"node_id": "O_CONTEXT", "evidence_id": "EV_CONTEXT", "available_at": context_at},
            {"node_id": "O_TRIGGER", "evidence_id": "EV_TRIGGER", "available_at": "PENDING"},
        ],
        "relations": [{"node_id": "R1", "relation": "CONTEXT_TRIGGER",
                       "input_ids": ["O_CONTEXT", "O_TRIGGER"], "available_at": "DECISION"}],
        "execution": {"entry_state": "DECISION", "pending_state": "PENDING",
                      "entry_action": "BID1_QUEUE",
                      "entry_lifecycle_policy": {"mode": mode,
                                                 "pending_observable_ids": list(pending_ids or [])}},
        "prediction": {"input_ids": ["R1"], "evaluation_state": "FILLED",
                       "target_type": "PATH_TARGET", "direction": "HIGHER"},
        "catalog_bindings": list(bindings),
    }


def known():
    return {"EV_CONTEXT", "EV_TRIGGER"}, {"vol_flow", "ofi_depth_10"}


def test_valid_hold_through_graph_has_stable_semantic_hash():
    evidence_ids, features = known()
    value = graph()
    report = M.validate(value, known_evidence_ids=evidence_ids, known_features=features)
    assert report["problems"] == []
    assert report["semantic_hash"] == M.semantic_hash(value)


def test_cancel_policy_requires_a_pending_observable():
    evidence_ids, features = known()
    report = M.validate(graph(mode="CANCEL_WHEN_ENTRY_SIGNAL_FALSE"),
                        known_evidence_ids=evidence_ids, known_features=features)
    assert any("pending_observable_ids" in problem for problem in report["problems"])


def test_cancel_policy_rejects_a_decision_only_observable():
    evidence_ids, features = known()
    report = M.validate(graph(mode="CANCEL_WHEN_ENTRY_SIGNAL_FALSE", pending_ids=["O_CONTEXT"]),
                        known_evidence_ids=evidence_ids, known_features=features)
    assert any("PENDING" in problem for problem in report["problems"])


def test_grounding_may_add_bindings_but_not_change_meaning():
    evidence_ids, features = known()
    source = graph(mode="CANCEL_WHEN_ENTRY_SIGNAL_FALSE", context_at="PENDING",
                   pending_ids=["O_CONTEXT", "O_TRIGGER"])
    grounded = graph(mode="CANCEL_WHEN_ENTRY_SIGNAL_FALSE", context_at="PENDING",
                     pending_ids=["O_CONTEXT", "O_TRIGGER"], bindings=[
        {"observable_id": "O_CONTEXT", "catalog_feature": "vol_flow"},
        {"observable_id": "O_TRIGGER", "catalog_feature": "ofi_depth_10"},
    ])
    report = M.validate_preservation(source, grounded, known_evidence_ids=evidence_ids,
                                    known_features=features)
    assert report["problems"] == []
    assert report["source_semantic_hash"] == report["grounded_semantic_hash"]


def test_grounding_cannot_change_lifecycle_policy():
    evidence_ids, features = known()
    source = graph()
    grounded = graph(context_at="PENDING", bindings=[
        {"observable_id": "O_CONTEXT", "catalog_feature": "vol_flow"},
        {"observable_id": "O_TRIGGER", "catalog_feature": "ofi_depth_10"},
    ])
    grounded["execution"]["entry_lifecycle_policy"] = {
        "mode": "CANCEL_WHEN_ENTRY_SIGNAL_FALSE",
        "pending_observable_ids": ["O_CONTEXT", "O_TRIGGER"]}
    report = M.validate_preservation(source, grounded, known_evidence_ids=evidence_ids,
                                    known_features=features)
    assert any("의미를 바꿨다" in problem for problem in report["problems"])
