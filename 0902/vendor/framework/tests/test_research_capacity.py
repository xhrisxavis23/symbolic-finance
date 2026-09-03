"""동기 batch 연구의 실행 slot 회수 계약."""

from __future__ import annotations

import os

from framework.modules import research_capacity


def test_same_process_reclaims_an_interrupted_prior_run_slot(tmp_path):
    prior = tmp_path / "runs" / "prior"
    prior.mkdir(parents=True)
    (prior / "research_progress.json").write_text(
        '{"state": "IN_PROGRESS", "owner_pid": ' + str(os.getpid()) + '}\n')

    assert research_capacity._can_reclaim({
        "owner_pid": os.getpid(), "output": str(prior),
    }) is True
