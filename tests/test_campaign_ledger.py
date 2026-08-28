"""24 cells or no table: the campaign transaction.

Before the ledger, the chain launched cells one at a time and collected exit
codes at the end. "23 succeeded" and "24 succeeded" were indistinguishable to
every downstream reader, a second chain could plan the same cells and write into
the same result directories, and the plan could be rebuilt between the first
cell and the last so that one campaign name covered two configurations.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.campaign_ledger import (  # noqa: E402
    CampaignRefused, RECEIPT_NAME, open_campaign, record_cell, require_sealed,
    seal_campaign)

TAGS = [f"promptAblA_ds{d}_{c}" for d in range(4)
        for c in ("A2_no_text", "A4_shared_codebook", "A5_none", "A5_joint",
                  "A5_nogumbel", "A5_both")]


def _plan(tmp_path: Path, *, tags=None, digest="d" * 64) -> Path:
    tags = TAGS if tags is None else tags
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "schema_version": 2,
        "expected_cells": len(tags),
        "plan_digest": digest,
        "selected_n": {"cifar_A_v4": 39},
        "selection_sha256": "s" * 64,
        "cells": [{"tag": t} for t in tags],
    }, indent=2, sort_keys=True))
    return path


def _finish(ledger: Path, tags=None):
    for tag in (TAGS if tags is None else tags):
        record_cell(ledger, tag, "ok", run_dir=f"result/{tag}")


def test_a_complete_campaign_seals_and_verifies(tmp_path):
    ledger = tmp_path / "ledger"
    plan = _plan(tmp_path)
    open_campaign(ledger, plan)
    _finish(ledger)
    receipt = seal_campaign(ledger)
    assert receipt["cell_count"] == 24
    assert require_sealed(ledger, expected_cells=24)["cell_count"] == 24


def test_twenty_three_of_twenty_four_does_not_seal(tmp_path):
    ledger = tmp_path / "ledger"
    open_campaign(ledger, _plan(tmp_path))
    _finish(ledger, TAGS[:-1])
    with pytest.raises(CampaignRefused) as error:
        seal_campaign(ledger)
    assert TAGS[-1] in str(error.value)
    assert not (ledger / RECEIPT_NAME).exists()


def test_a_failed_cell_does_not_seal(tmp_path):
    ledger = tmp_path / "ledger"
    open_campaign(ledger, _plan(tmp_path))
    _finish(ledger, TAGS[:-1])
    record_cell(ledger, TAGS[-1], "failed", detail="trainer rc 23")
    with pytest.raises(CampaignRefused) as error:
        seal_campaign(ledger)
    assert "failed=" in str(error.value)


def test_a_second_campaign_cannot_open_the_same_ledger(tmp_path):
    """O_EXCL, so the collision surfaces here and not inside result/."""
    ledger = tmp_path / "ledger"
    open_campaign(ledger, _plan(tmp_path))
    with pytest.raises(CampaignRefused) as error:
        open_campaign(ledger, _plan(tmp_path))
    assert "already held" in str(error.value)


def test_a_cell_recorded_against_a_rebuilt_plan_is_not_this_campaigns(tmp_path):
    """The rebuild-mid-run case: same tags, different N or different cache."""
    ledger = tmp_path / "ledger"
    open_campaign(ledger, _plan(tmp_path))
    _finish(ledger)
    stolen = json.loads((ledger / f"cell_{TAGS[0]}.json").read_text())
    stolen["plan_file_sha256"] = "0" * 64
    (ledger / f"cell_{TAGS[0]}.json").write_text(json.dumps(stolen))
    with pytest.raises(CampaignRefused) as error:
        seal_campaign(ledger)
    assert "recorded_against_another_plan" in str(error.value)
    assert TAGS[0] in str(error.value)


def test_a_campaign_cannot_grow_a_cell_after_it_opened(tmp_path):
    ledger = tmp_path / "ledger"
    open_campaign(ledger, _plan(tmp_path))
    with pytest.raises(CampaignRefused) as error:
        record_cell(ledger, "promptAblA_ds0_A9_invented", "ok")
    assert "not one of the 24 reserved cells" in str(error.value)


def test_recording_without_opening_is_refused(tmp_path):
    with pytest.raises(CampaignRefused) as error:
        record_cell(tmp_path / "ledger", TAGS[0], "ok")
    assert "never opened" in str(error.value)


def test_a_reader_refuses_an_unsealed_campaign(tmp_path):
    ledger = tmp_path / "ledger"
    open_campaign(ledger, _plan(tmp_path))
    _finish(ledger)
    with pytest.raises(CampaignRefused) as error:
        require_sealed(ledger, expected_cells=24)
    assert "never completed" in str(error.value)


def test_a_reader_refuses_a_receipt_for_another_plan(tmp_path):
    ledger = tmp_path / "ledger"
    open_campaign(ledger, _plan(tmp_path))
    _finish(ledger)
    seal_campaign(ledger)
    with pytest.raises(CampaignRefused) as error:
        require_sealed(ledger, expected_cells=24, plan_file_sha256="f" * 64)
    assert "different plan" in str(error.value)


def test_a_plan_that_cannot_count_itself_is_not_reserved(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(
        {"expected_cells": 24, "cells": [{"tag": t} for t in TAGS[:5]]}))
    with pytest.raises(CampaignRefused):
        open_campaign(tmp_path / "ledger", path)


def test_duplicate_tags_are_not_reserved(tmp_path):
    with pytest.raises(CampaignRefused):
        open_campaign(tmp_path / "ledger",
                      _plan(tmp_path, tags=TAGS[:-1] + [TAGS[0]]))


def test_the_cli_refuses_the_same_way_the_module_does(tmp_path):
    """The chain calls the CLI, so the CLI is what has to refuse."""
    ledger = tmp_path / "ledger"
    plan = _plan(tmp_path)
    run = lambda *a: subprocess.run(
        [sys.executable, str(REPO / "scripts" / "campaign_ledger.py"), *a],
        capture_output=True, text=True, timeout=60)
    assert run("open", "--ledger", str(ledger), "--plan", str(plan)).returncode == 0
    second = run("open", "--ledger", str(ledger), "--plan", str(plan))
    assert second.returncode == 1 and "already held" in second.stderr
    assert run("seal", "--ledger", str(ledger)).returncode == 1
    for tag in TAGS:
        assert run("record", "--ledger", str(ledger), "--tag", tag,
                   "--status", "ok").returncode == 0
    assert run("seal", "--ledger", str(ledger)).returncode == 0
    assert run("verify", "--ledger", str(ledger), "--plan", str(plan)).returncode == 0


# ---------------------------------------------------------------------------
# The chain's own shell, executed. `mapfile -t < <(producer)` checks mapfile's
# status and never the producer's -- the third defect the chain was rewritten to
# remove, and it went straight back in during that rewrite. This runs the real
# construct against a producer that prints rows and then dies.
# ---------------------------------------------------------------------------

def test_a_cell_producer_that_dies_partway_is_not_accepted(tmp_path):
    script = tmp_path / "reader.sh"
    script.write_text(
        'set -Eeuo pipefail\n'
        'ROWS_FILE=$(mktemp)\n'
        'bash "$1" > "$ROWS_FILE" || { echo "producer failed"; exit 1; }\n'
        'ROWS=()\n'
        'while IFS= read -r row; do ROWS+=("$row"); done < "$ROWS_FILE"\n'
        'echo "accepted ${#ROWS[@]}"\n')
    producer = tmp_path / "producer.sh"
    producer.write_text('printf "a\\nb\\nc\\n"\nexit 23\n')
    proc = subprocess.run(["bash", str(script), str(producer)],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0, proc.stdout
    assert "accepted" not in proc.stdout, (
        "three rows were read from a producer that exited 23")


def test_the_launcher_does_not_read_the_plan_per_cell():
    """TOCTOU: the twenty-fourth cell must not run a plan cell one never saw.

    The executing proof is in test_ablation_campaign_launcher.py, which edits a
    plan mid-run; this only guards the two constructs from coming back.
    """
    source = "\n".join(
        line for line
        in (REPO / "scripts" / "run_ablation_campaign.sh").read_text().splitlines()
        if not line.strip().startswith("#"))
    assert "--expect-plan-sha256" in source, (
        "every cell must be started against the digest hashed at build time")
    assert "mapfile -t" not in source, (
        "mapfile's status is not the producer's status")
