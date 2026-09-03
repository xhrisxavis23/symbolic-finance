"""Sandbox의 Agent·직접 Evidence 후보를 다음 Agent의 중복 방지용 library로 만든다."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from ..config import clean, sha256_json
from ..contracts.artifacts import ArtifactError, create_artifact, read_artifact, write_artifact


SCHEMA_VERSION = "candidate_library.v3"
CACHE_SCHEMA_VERSION = "candidate_library_cache.v3"
ARTIFACT_NAME = "candidate_library_artifact.json"
SANDBOX_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = SANDBOX_ROOT / "cache" / "candidate_library"
TITLE_LIMIT = 80
CLAIM_LIMIT = 120
PROFIT_LIMIT = 90
TEST_LIMIT = 100


def build(output: Path, *, runs_root: Path | None = None,
          exclude_artifact_ids: set[str] | None = None,
          exclude_descendants_of: set[str] | None = None,
          direct_evidence_id: str | None = None,
          direct_profile: Mapping[str, Any] | None = None,
          cache_root: Path | None = None) -> dict[str, Any]:
    """현재 output 이전의 모든 Sandbox Hypothesis artifact를 결정적 순서로 모은다."""
    runs_root = Path(runs_root or (SANDBOX_ROOT / "runs"))
    excluded = {str(value) for value in (exclude_artifact_ids or set())}
    excluded_lineages = {str(value) for value in (exclude_descendants_of or set())}
    expected_direct_profile = clean(dict(direct_profile)) if direct_profile is not None else None
    scope = {
        "schema": CACHE_SCHEMA_VERSION,
        "runs_root": str(runs_root.resolve()),
        "direct_evidence_id": str(direct_evidence_id) if direct_evidence_id is not None else None,
        "direct_profile": expected_direct_profile,
        "excluded_source_artifacts": sorted(excluded),
        "excluded_candidate_lineages": sorted(excluded_lineages),
    }
    inventory = _source_inventory(runs_root)
    cache_path = _cache_path(cache_root or CACHE_ROOT, scope)
    cached = _read_cache(cache_path, scope=scope)
    records = (list(cached["records"])
               if cached is not None
               and cached.get("source_fingerprint") == inventory["fingerprint"] else None)
    if records is None and cached is not None:
        records = _extend_append_only_history(
            cached, inventory, excluded=excluded, excluded_lineages=excluded_lineages)
    if records is None:
        records = _collect_records(
            inventory,
            excluded=excluded,
            excluded_lineages=excluded_lineages,
            direct_evidence_id=direct_evidence_id,
            expected_direct_profile=expected_direct_profile,
        )
        _write_cached_records(cache_path, scope=scope,
                              source_fingerprint=inventory["fingerprint"],
                              source_manifest=inventory["manifest"],
                              records=records)
    elif cached is None or cached.get("source_fingerprint") != inventory["fingerprint"]:
        _write_cached_records(cache_path, scope=scope,
                              source_fingerprint=inventory["fingerprint"],
                              source_manifest=inventory["manifest"],
                              records=records)
    return _write_library(
        output,
        records=records,
        direct_evidence_id=direct_evidence_id,
        expected_direct_profile=expected_direct_profile,
        excluded=excluded,
        excluded_lineages=excluded_lineages,
    )


def extend_snapshot(output: Path, source: Path,
                    hypothesis_paths: list[Path]) -> dict[str, Any]:
    """고정 history 위에 현재 loop가 이미 만든 Agent 가설만 덧붙인다.

    전역 ``runs/``를 다시 읽지 않는다. 이 함수는 같은 loop의 이전 generation/lens가
    만든 hypothesis artifact만 읽어, 고정 snapshot의 novelty 경계를 유지하면서도
    sibling 가설을 새 설명처럼 되풀이하지 않게 한다.
    """
    base = read_artifact(Path(source), kind="candidate_library")
    saved = base["payload"]
    records = [dict(row) for row in saved.get("records") or [] if isinstance(row, Mapping)]
    known = {(str(row.get("candidate_kind")), str(row.get("source_artifact")),
              str(row.get("hypothesis_id"))) for row in records}
    added_ids: list[str] = []
    for path in sorted({Path(item).resolve() for item in hypothesis_paths}):
        try:
            hypothesis_set = read_artifact(path, kind="hypothesis_set")
        except (ArtifactError, OSError, ValueError):
            continue
        for hypothesis in ((hypothesis_set["payload"].get("payload") or {}).get("hypotheses") or []):
            if not isinstance(hypothesis, Mapping):
                continue
            key = ("AGENT_HYPOTHESIS", str(hypothesis_set["artifact_id"]),
                   str(hypothesis.get("hypothesis_id")))
            if key in known:
                continue
            known.add(key)
            records.append(_record(path, hypothesis_set, hypothesis))
            added_ids.append(str(hypothesis_set["artifact_id"]))
    records.sort(key=lambda row: (row["clusters"], row["source_artifact"], row["hypothesis_id"]))
    artifact = create_artifact("candidate_library", {
        "schema": SCHEMA_VERSION,
        "scope": saved.get("scope"),
        "direct_evidence_id": saved.get("direct_evidence_id"),
        "direct_profile": saved.get("direct_profile"),
        "excluded_source_artifacts": list(saved.get("excluded_source_artifacts") or []),
        "excluded_candidate_lineages": list(saved.get("excluded_candidate_lineages") or []),
        "records": records,
        "loop_snapshot_extension": {
            "base_candidate_library": str(base["artifact_id"]),
            "added_hypothesis_sets": sorted(set(added_ids)),
            "global_history_reopened": False,
        },
        "note": (
            "고정 historical candidate library에 현재 Discovery loop가 이미 만든 Agent "
            "가설만 덧붙인 snapshot이다. 전역 run history나 PnL은 다시 읽지 않는다."),
    }, parents={"base_candidate_library": str(base["artifact_id"]),
                **{f"loop_hypothesis_{index}": artifact_id
                   for index, artifact_id in enumerate(sorted(set(added_ids)), 1)}})
    write_artifact(Path(output) / ARTIFACT_NAME, artifact)
    return artifact


def rebind_direct_evidence(output: Path, source: Path, *, direct_evidence_id: str,
                           direct_profile: Mapping[str, Any]) -> dict[str, Any]:
    """같은 저장 Profile의 frozen history를 다른 immutable Evidence ID에 연결한다.

    Direct 후보의 범위는 build 시에도 Evidence ID보다 동일 Feature Profile을 우선한다.
    따라서 Profile이 정확히 같을 때만 direct-evidence metadata를 target Evidence로
    바꾼 child snapshot을 만든다. records는 새로 수집하지 않는다.
    """
    base = read_artifact(Path(source), kind="candidate_library")
    saved = base["payload"]
    expected_profile = clean(dict(direct_profile))
    if saved.get("direct_profile") != expected_profile:
        raise ValueError("candidate snapshot의 Direct Feature Profile이 target과 다르다")
    artifact = create_artifact("candidate_library", {
        "schema": SCHEMA_VERSION,
        "scope": saved.get("scope"),
        "direct_evidence_id": str(direct_evidence_id),
        "direct_profile": expected_profile,
        "excluded_source_artifacts": list(saved.get("excluded_source_artifacts") or []),
        "excluded_candidate_lineages": list(saved.get("excluded_candidate_lineages") or []),
        "records": [dict(row) for row in saved.get("records") or [] if isinstance(row, Mapping)],
        "rebound_direct_evidence": {
            "base_candidate_library": str(base["artifact_id"]),
            "source_direct_evidence_id": saved.get("direct_evidence_id"),
            "target_direct_evidence_id": str(direct_evidence_id),
            "profile_matched": True,
            "global_history_reopened": False,
        },
        "note": (
            "같은 저장 Feature Profile의 frozen candidate history를 target Evidence ID에 "
            "provenance-preserving rebind한 snapshot이다. records를 다시 수집하지 않는다."),
    }, parents={"base_candidate_library": str(base["artifact_id"]),
                "target_evidence_package": str(direct_evidence_id)})
    write_artifact(Path(output) / ARTIFACT_NAME, artifact)
    return artifact


def _write_library(output: Path, *, records: list[dict[str, Any]],
                   direct_evidence_id: str | None,
                   expected_direct_profile: Mapping[str, Any] | None,
                   excluded: set[str], excluded_lineages: set[str]) -> dict[str, Any]:
    artifact = create_artifact("candidate_library", {
        "schema": SCHEMA_VERSION,
        "scope": "Sandbox hypothesis artifacts and direct Evidence candidates",
        "direct_evidence_id": str(direct_evidence_id) if direct_evidence_id is not None else None,
        "direct_profile": expected_direct_profile,
        "excluded_source_artifacts": sorted(excluded),
        "excluded_candidate_lineages": sorted(excluded_lineages),
        "records": records,
        "note": (
            "이 library는 과거 Agent 가설과 직접 Evidence entry logic의 중복을 막는 비교 입력이다. "
            "직접 Evidence 후보는 mechanism을 주장하지 않으며, 어느 후보도 수익성·진실성·"
            "Validation 지지를 뜻하지 않는다."
        ),
    })
    write_artifact(Path(output) / ARTIFACT_NAME, artifact)
    return artifact


def _collect_records(inventory: Mapping[str, Any], *, excluded: set[str],
                     excluded_lineages: set[str], direct_evidence_id: str | None,
                     expected_direct_profile: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """한 번 수집한 파일 목록만 써서 library record를 만든다.

    이전 구현은 direct candidate artifact 하나마다 run 전체를 다시 ``rglob`` 했다.
    후보가 늘수록 같은 history를 N번 읽는 구조라 loop의 다음 세대로 갈수록 느려졌다.
    """
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    direct_records: dict[tuple[str, ...], dict[str, Any]] = {}
    evidence_profiles = _evidence_profile_index_from_paths(inventory["evidence"])
    pre_validation_profiles = _pre_validation_profile_index(inventory["pre_validation"])
    opened_by_direct_root = _direct_evaluation_index(inventory)
    for path in inventory["hypothesis"]:
        try:
            artifact = read_artifact(path, kind="hypothesis_set")
        except (ArtifactError, OSError, ValueError):
            continue
        if str(artifact["artifact_id"]) in excluded:
            continue
        if str((artifact.get("parents") or {}).get("source_hypothesis_set") or "") in excluded_lineages:
            continue
        payload = artifact["payload"]
        for hypothesis in ((payload.get("payload") or {}).get("hypotheses") or []):
            if isinstance(hypothesis, Mapping):
                key = ("agent", str(artifact["artifact_id"]), str(hypothesis.get("hypothesis_id")))
                if key not in seen:
                    seen.add(key)
                    records.append(_record(path, artifact, hypothesis,
                                           pre_validation_profiles=pre_validation_profiles))
    # Direct Evidence도 실제 Specification까지 이어지는 Sandbox 후보이다. 이들을
    # Agent history에서 빼면 Agent가 동일 entry logic을 새 mechanism처럼 다시
    # 제안할 수 있다. 다만 직접 관측 후보는 아직 financial mechanism을 주장하지
    # 않으므로, 그 사실과 entry fingerprint를 함께 보존한다.
    for path in inventory["direct"]:
        try:
            artifact = read_artifact(path, kind="direct_evidence_candidate_set")
        except (ArtifactError, OSError, ValueError):
            continue
        profile = _direct_profile_for(
            path, artifact=artifact, evidence_profiles=evidence_profiles,
            pre_validation_profiles=pre_validation_profiles)
        # 동일 Profile 비교가 가능하면 artifact ID보다 그것을 정본으로 쓴다.
        # Evidence는 같은 저장 Feature Profile에서 다시 만들 수 있으므로 ID가
        # 달라져도 같은 entry contract를 새 후보로 열면 안 된다.
        in_scope = (profile == expected_direct_profile
                    if expected_direct_profile is not None else
                    (direct_evidence_id is None
                     or str(artifact.get("parents", {}).get("evidence_package"))
                     == str(direct_evidence_id)))
        if not in_scope:
            continue
        opened = opened_by_direct_root.get(Path(path).parents[1], set())
        for candidate in artifact["payload"].get("candidates") or []:
            if isinstance(candidate, Mapping):
                # 같은 Evidence를 여러 Direct workflow가 재사용하면 artifact id는
                # 달라도 같은 entry logic이 반복된다. library에는 한 번만 남긴다.
                key = ("direct", str(candidate.get("hypothesis_id")),
                       str(candidate.get("entry_logic_fingerprint")))
                record = _record_direct(
                    path, artifact, candidate, profile=profile,
                    evaluation=_direct_evaluation_status(candidate, opened_ids=opened))
                if key not in seen:
                    seen.add(key)
                    direct_records[key] = record
                    records.append(record)
                elif (record["evaluation_opened"]
                      and not direct_records[key]["evaluation_opened"]):
                    # 같은 logic의 관찰 artifact가 먼저 나와도, 나중에 실제 replay로
                    # 열린 lineage가 있으면 중복 제외 판단은 그 사실을 따라야 한다.
                    direct_records[key].update(record)
    records.sort(key=lambda row: (row["clusters"], row["source_artifact"], row["hypothesis_id"]))
    return records


def _source_inventory(runs_root: Path) -> dict[str, Any]:
    """후보 library에 영향을 주는 파일을 한 번만 순회해 기록한다."""
    groups: dict[str, list[Path]] = {
        "evidence": [], "hypothesis": [], "direct": [],
        "entry_refinement_artifact": [], "entry_refinement": [], "pre_validation": [],
    }
    names = {
        "evidence_artifact.json": "evidence",
        "hypothesis_artifact.json": "hypothesis",
        "direct_evidence_candidates_artifact.json": "direct",
        "entry_refinement_artifact.json": "entry_refinement_artifact",
        "entry_refinement.json": "entry_refinement",
        "pre_validation_run.json": "pre_validation",
    }
    manifest: list[dict[str, Any]] = []
    if runs_root.is_dir():
        for directory, _, filenames in os.walk(runs_root):
            parent = Path(directory)
            for name in filenames:
                group = names.get(name)
                if group is None:
                    continue
                path = parent / name
                try:
                    stat = path.stat()
                except OSError:
                    continue
                groups[group].append(path)
                manifest.append({
                    "path": str(path.relative_to(runs_root)),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "size": int(stat.st_size),
                })
    for paths in groups.values():
        paths.sort()
    manifest.sort(key=lambda row: row["path"])
    return {**groups, "runs_root": runs_root, "manifest": manifest,
            "fingerprint": sha256_json(manifest)}


def _cache_path(cache_root: Path, scope: Mapping[str, Any]) -> Path:
    return Path(cache_root) / f"{sha256_json(scope)[:24]}.json"


def _read_cache(path: Path, *, scope: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        cache = read_artifact(path, kind="candidate_library_cache")
    except (ArtifactError, OSError, ValueError):
        return None
    payload = cache["payload"]
    records, manifest = payload.get("records"), payload.get("source_manifest")
    if (payload.get("schema") != CACHE_SCHEMA_VERSION
            or payload.get("scope") != clean(dict(scope))
            or not isinstance(records, list)
            or not all(isinstance(row, Mapping) for row in records)
            or not isinstance(manifest, list)
            or not all(isinstance(row, Mapping) for row in manifest)):
        return None
    return {"source_fingerprint": payload.get("source_fingerprint"),
            "source_manifest": [dict(row) for row in manifest],
            "records": [dict(row) for row in records]}


def _write_cached_records(path: Path, *, scope: Mapping[str, Any],
                          source_fingerprint: str,
                          source_manifest: list[dict[str, Any]],
                          records: list[dict[str, Any]]) -> None:
    cache = create_artifact("candidate_library_cache", {
        "schema": CACHE_SCHEMA_VERSION,
        "scope": dict(scope),
        "source_fingerprint": source_fingerprint,
        "source_manifest": source_manifest,
        "records": records,
    })
    write_artifact(path, cache)


def _extend_append_only_history(cached: Mapping[str, Any], inventory: Mapping[str, Any], *,
                                excluded: set[str],
                                excluded_lineages: set[str]) -> list[dict[str, Any]] | None:
    """새 hypothesis/pre-validation 기록만 추가됐을 때 기존 parse 결과를 이어 붙인다.

    loop retry는 직전 Agent artifact와 terminal pre-validation manifest만 더하는 경우가
    대부분이다. 그때 수백 개의 과거 Agent 응답을 다시 무결성 hash할 이유는 없다.
    Direct candidate·Evidence·Entry Refinement가 바뀌면 eligibility가 달라질 수 있으므로
    보수적으로 전체 collection으로 돌아간다.
    """
    old = {str(row.get("path")): (row.get("mtime_ns"), row.get("size"))
           for row in cached.get("source_manifest") or []}
    current = {str(row.get("path")): (row.get("mtime_ns"), row.get("size"))
               for row in inventory["manifest"]}
    if any(current.get(path) != marker for path, marker in old.items()):
        return None
    additions = sorted(set(current) - set(old))
    allowed = {"hypothesis_artifact.json", "pre_validation_run.json"}
    if not additions or any(Path(path).name not in allowed for path in additions):
        return None
    records = [dict(row) for row in cached.get("records") or []]
    # A new pre-validation manifest gives a newly-added hypothesis its profile. Refreshing
    # only this compact record field avoids reopening the large hypothesis payloads.
    for record in records:
        if record.get("candidate_kind") == "AGENT_HYPOTHESIS":
            record["clusters"] = list(_profile_for(Path(str(record["source_path"]))).get("clusters") or [])
    known = {(str(row.get("source_artifact")), str(row.get("hypothesis_id")))
             for row in records if row.get("candidate_kind") == "AGENT_HYPOTHESIS"}
    added_paths = {str(Path(path)) for path in additions if Path(path).name == "hypothesis_artifact.json"}
    for path in inventory["hypothesis"]:
        try:
            relative = str(path.relative_to(_inventory_root(inventory)))
        except ValueError:
            return None
        if relative not in added_paths:
            continue
        try:
            artifact = read_artifact(path, kind="hypothesis_set")
        except (ArtifactError, OSError, ValueError):
            continue
        if str(artifact["artifact_id"]) in excluded:
            continue
        if str((artifact.get("parents") or {}).get("source_hypothesis_set") or "") in excluded_lineages:
            continue
        for hypothesis in ((artifact["payload"].get("payload") or {}).get("hypotheses") or []):
            if not isinstance(hypothesis, Mapping):
                continue
            key = (str(artifact["artifact_id"]), str(hypothesis.get("hypothesis_id")))
            if key not in known:
                known.add(key)
                records.append(_record(path, artifact, hypothesis))
    records.sort(key=lambda row: (row["clusters"], row["source_artifact"], row["hypothesis_id"]))
    return records


def _inventory_root(inventory: Mapping[str, Any]) -> Path:
    # The relative paths in a manifest have no root. Every non-empty inventory path shares
    # its runs root through the caller's `_source_inventory` field below.
    root = inventory.get("runs_root")
    if root is None:
        raise ValueError("candidate library inventory에 runs_root가 없다")
    return Path(str(root))


def _pre_validation_profile_index(paths: list[Path]) -> dict[Path, Mapping[str, Any]]:
    profiles: dict[Path, Mapping[str, Any]] = {}
    for path in paths:
        try:
            run = read_artifact(path, kind="pre_validation_run")
        except (ArtifactError, OSError, ValueError):
            continue
        profile = (run["payload"].get("request") or {}).get("profile") or {}
        if isinstance(profile, Mapping):
            profiles[path.parent] = clean(dict(profile))
    return profiles


def _profile_from_index(path: Path, profiles: Mapping[Path, Mapping[str, Any]]) -> Mapping[str, Any]:
    matched: tuple[int, Mapping[str, Any]] | None = None
    for root, profile in profiles.items():
        try:
            path.relative_to(root)
        except ValueError:
            continue
        candidate = (len(root.parts), profile)
        if matched is None or candidate[0] > matched[0]:
            matched = candidate
    return matched[1] if matched is not None else {}


def _direct_evaluation_index(inventory: Mapping[str, Any]) -> dict[Path, set[str]]:
    """모든 Direct root의 실제 Entry Refinement 개시 여부를 작은 marker로 읽는다.

    unit별 ``entry_refinement.json``은 경로 자체에 후보 ID가 있다. 이전 방식처럼
    ledger가 수십 MB인 parent artifact를 전부 JSON parse하면, 중복 방지 library를
    만들기만 해도 GB 단위의 과거 ledger를 매번 읽게 된다. unit marker가 없는
    소수 legacy root만 parent artifact로 보완한다.
    """
    roots = sorted({Path(path).parents[1] for path in inventory["direct"]})
    opened = {root: set() for root in roots}
    if not roots:
        return opened
    roots_with_unit_markers: set[Path] = set()
    for path in inventory["entry_refinement"]:
        target = _containing_direct_root(path, roots)
        hypothesis_id = _unit_hypothesis_id(path)
        if target is not None and hypothesis_id:
            opened[target].add(hypothesis_id)
            roots_with_unit_markers.add(target)
    for path in inventory["entry_refinement_artifact"]:
        target = _containing_direct_root(path, roots)
        if target is None or target in roots_with_unit_markers:
            continue
        try:
            saved = read_artifact(path, kind="entry_refinement_set")
        except (ArtifactError, OSError, ValueError):
            continue
        units = (saved.get("payload") or {}).get("units") or {}
        for hypothesis_id, unit in units.items():
            if (isinstance(unit, Mapping)
                    and str(unit.get("state") or "") != "IMPLEMENTATION_NOT_READY"):
                opened[target].add(str(hypothesis_id))
    return opened


def _unit_hypothesis_id(path: Path) -> str:
    """``.../units/<hypothesis_id>/.../entry_refinement.json``의 ID를 경로에서 얻는다."""
    parts = path.parts
    try:
        index = parts.index("units")
    except ValueError:
        return ""
    return str(parts[index + 1]) if index + 1 < len(parts) else ""


def _containing_direct_root(path: Path, roots: list[Path]) -> Path | None:
    matches: list[Path] = []
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        matches.append(root)
    return max(matches, key=lambda root: len(root.parts)) if matches else None


def _record(path: Path, artifact: Mapping[str, Any], hypothesis: Mapping[str, Any], *,
            pre_validation_profiles: Mapping[Path, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    profile = (_profile_from_index(path, pre_validation_profiles)
               if pre_validation_profiles is not None else _profile_for(path))
    return {
        "candidate_kind": "AGENT_HYPOTHESIS",
        "source_artifact": str(artifact["artifact_id"]),
        "source_path": str(path),
        "source_state": str(artifact["payload"].get("state")),
        "clusters": list(profile.get("clusters") or []),
        "hypothesis_id": str(hypothesis.get("hypothesis_id")),
        # Agent에 주는 historical library는 중복 비교용 fingerprint다. 원문 전체는
        # 개별 artifact에 이미 보존돼 있고, 그대로 prompt에 싣으면 후보 수가 늘 때
        # OS argument 한도를 넘겨 Agent 호출 자체가 실패한다.
        "title": _clip(hypothesis.get("title"), TITLE_LIMIT),
        "hypothesis_structure": _clip(hypothesis.get("hypothesis_structure"), TITLE_LIMIT),
        "evidence_basis": [{key: item.get(key) for key in ("family", "representative_feature")}
                           for item in (hypothesis.get("evidence_basis") or [])[:4]],
        "mechanistic_claim": _clip(hypothesis.get("mechanistic_claim"), CLAIM_LIMIT),
        "source_of_profit": _clip(hypothesis.get("source_of_profit"), PROFIT_LIMIT),
        "mechanism_tests": [_test_fingerprint(item)
                            for item in (hypothesis.get("mechanism_tests") or [])[:1]],
    }


def _record_direct(path: Path, artifact: Mapping[str, Any], candidate: Mapping[str, Any], *,
                   profile: Mapping[str, Any], evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """직접 관측 후보를 mechanism 없는 기존 entry logic으로 library에 보존한다."""
    # fingerprint는 실행 계약의 identity다. 여기서 자르면 이후의 중복 제외가
    # 문장 일부 비교가 되어 버리므로, Agent에 보일 prose와 달리 원문을 보존한다.
    entry_logic = str(candidate.get("entry_logic_fingerprint") or "")
    return {
        "candidate_kind": "DIRECT_EVIDENCE_CANDIDATE",
        "source_artifact": str(artifact["artifact_id"]),
        "source_path": str(path),
        "source_state": str(artifact["payload"].get("state")),
        "clusters": list(profile.get("clusters") or []),
        "hypothesis_id": str(candidate.get("hypothesis_id")),
        "title": _clip(candidate.get("title"), TITLE_LIMIT),
        "hypothesis_structure": _clip(candidate.get("source_type"), TITLE_LIMIT),
        "evidence_basis": [dict(item) for item in (candidate.get("evidence_basis") or [])
                           if isinstance(item, Mapping)][:4],
        "entry_logic_fingerprint": entry_logic,
        "evaluation_opened": bool(evaluation["opened"]),
        "evaluation_state": str(evaluation["state"]),
        "mechanistic_claim": "직접 관측 후보: 경제적 mechanism은 아직 해석하지 않았다.",
        "source_of_profit": ("미확정. 기존 entry logic은 " + entry_logic) if entry_logic else "미확정",
        "mechanism_tests": [{
            "test_type": "DIRECT_ENTRY_LOGIC",
            "claim": "같은 entry logic을 단순 문장·임계값 변경으로 새 가설로 세지 않는다.",
        }],
    }


def _direct_evaluation_status(candidate: Mapping[str, Any], *,
                              opened_ids: set[str]) -> dict[str, Any]:
    """후보를 본 것과 같은 entry를 실제 Discovery replay에 보낸 것을 구분한다.

    Direct candidate artifact는 Agent의 비교용으로는 보존돼야 한다. 하지만 Specification
    출력만 남고 Entry Refinement가 한 번도 열리지 않은 후보까지 다음 Direct 실행에서
    제외하면, 관찰만 했던 후보가 영구히 Search에 못 간다.
    """
    hypothesis_id = str(candidate.get("hypothesis_id") or "")
    # 정상 Direct workflow는 candidate id별 unit을 artifact에 보존한다. 이 시점부터는
    # 같은 entry logic을 다시 Search할 이유가 없다. 단순 specification/implementation
    # artifact와 EXECUTION_EQUIVALENT는 여기 포함하지 않는다. 후자는 replay가 아니라
    # 구현 계약 비교일 뿐이라, 새 workflow가 명시적으로 같은 계약임을 기록하게 둔다.
    if hypothesis_id in opened_ids:
        return {"opened": True, "state": "ENTRY_REFINEMENT"}
    return {"opened": False, "state": "OBSERVED_ONLY"}


def _clip(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _test_fingerprint(item: Mapping[str, Any]) -> dict[str, str]:
    return {"test_type": _clip(item.get("test_type"), TITLE_LIMIT),
            "claim": _clip(item.get("claim"), TEST_LIMIT)}


def _profile_for(path: Path) -> Mapping[str, Any]:
    for parent in (path.parent, *path.parents):
        candidate = parent / "pre_validation_run.json"
        if candidate.is_file():
            try:
                artifact = read_artifact(candidate, kind="pre_validation_run")
                return ((artifact["payload"].get("request") or {}).get("profile") or {})
            except (ArtifactError, OSError, ValueError):
                return {}
    return {}


def _evidence_profile_index_from_paths(paths: list[Path]) -> dict[str, Mapping[str, Any]]:
    """Evidence artifact ID별 저장 Profile selection을 한 번만 읽는다."""
    profiles: dict[str, Mapping[str, Any]] = {}
    for path in paths:
        try:
            evidence = read_artifact(path, kind="evidence_package")
        except (ArtifactError, OSError, ValueError):
            continue
        selection = evidence["payload"].get("selection") or {}
        if isinstance(selection, Mapping):
            profiles[str(evidence["artifact_id"])] = clean(dict(selection))
    return profiles


def _direct_profile_for(path: Path, *, artifact: Mapping[str, Any],
                        evidence_profiles: Mapping[str, Mapping[str, Any]],
                        pre_validation_profiles: Mapping[Path, Mapping[str, Any]] | None = None
                        ) -> Mapping[str, Any]:
    """Direct 후보의 sibling 또는 부모 Evidence에서 Profile selection을 복원한다."""
    evidence_path = Path(path).parents[1] / "01_evidence" / "evidence_artifact.json"
    if evidence_path.is_file():
        try:
            evidence = read_artifact(evidence_path, kind="evidence_package")
            selection = evidence["payload"].get("selection") or {}
            if isinstance(selection, Mapping):
                return clean(dict(selection))
        except (ArtifactError, OSError, ValueError):
            pass
    evidence_id = str((artifact.get("parents") or {}).get("evidence_package") or "")
    if evidence_id in evidence_profiles:
        return evidence_profiles[evidence_id]
    return (_profile_from_index(path, pre_validation_profiles)
            if pre_validation_profiles is not None else _profile_for(path))
