"""A finding we could not read must never be filed as a verdict we did read.

Observed 2026-08-27 in run `…__p2_SMOKE`: an investigator diagnosed `tools` as
the culprit and proposed a concrete fix, but while enumerating command names
inside `suggested_change` it wrote a bare `"` (`… scdaemon", etc.)`), which ends
the JSON string early and invalidates the whole object. `parse_contrast_finding`
caught the ValueError and returned `{"is_culprit": False, "note": …}` — byte-for
-byte the shape of a genuine "no, that module is innocent" verdict. It was the
only culprit in the batch, so the window produced no proposal at all and the
loss was invisible in every log.

Two separate properties are at stake and they need separate tests:

1. **Recover what is recoverable.** One stray quote invalidated 8775 otherwise
   well-formed characters. `is_culprit` sat three lines above the damage.
2. **Never fake a verdict.** When recovery genuinely fails there is no verdict,
   and the record has to say so — a silent `is_culprit: False` is a fabricated
   observation, and fabricated observations are exactly what the evidence
   channel exists to prevent.
"""

import json

import pytest

from harbor.agents.terminus_2_modular.self_evo import backlog as _backlog
from harbor.agents.terminus_2_modular.self_evo.trajectory_analysis import (
    PARSE_FAILED,
    PARSE_OK,
    PARSE_SALVAGED,
    loads_finding_block,
    parse_contrast_finding,
)

pytestmark = pytest.mark.unit


# The real block from the run, trimmed in the middle of the command list but
# keeping the exact defect: a bare `"` after `scdaemon`.
REAL_CORRUPT_BLOCK = """{
  "task": "draft_dp_0c5cabc8_haitian",
  "is_culprit": true,
  "locked_module": "tools",
  "divergence": "The failing roll recorded two `parse_llm_response` calls with `PARSE-ERROR`, returning 0 commands and `task_complete=False`.",
  "would_change_outcome": "Yes. The current `parse_llm_response` delegates entirely to `TerminusXMLPlainParser` with zero fallback.",
  "fixable_now": true,
  "suggested_change": "Edit `tools/baseline.py` to add a regex fallback: extract lines starting with common tool names (ls, cat, sed, grep, gpg, ssh-keygen, scdaemon", etc.) and treat each as a command; then scan for \\"task complete\\" in any capitalization.",
  "other_module": null
}"""


def _traj_with(tmp_path, block: str, tag: str = "contrast_finding"):
    traj = tmp_path / "trajectory.json"
    traj.write_text(
        json.dumps(
            {"steps": [{"source": "agent", "message": f"<{tag}>{block}</{tag}>"}]}
        )
    )
    return traj


def test_the_real_block_is_genuinely_invalid_json(tmp_path):
    """Guards the fixture: if this ever parses, the tests below prove nothing."""
    with pytest.raises(ValueError):
        json.loads(REAL_CORRUPT_BLOCK)


def test_a_stray_quote_does_not_erase_a_culprit_verdict(tmp_path):
    got = parse_contrast_finding(_traj_with(tmp_path, REAL_CORRUPT_BLOCK), "t")
    assert got["is_culprit"] is True


def test_a_salvaged_finding_keeps_the_fields_before_the_damage(tmp_path):
    got = parse_contrast_finding(_traj_with(tmp_path, REAL_CORRUPT_BLOCK), "t")
    assert got["locked_module"] == "tools"
    assert got["fixable_now"] is True
    assert "PARSE-ERROR" in got["divergence"]


def test_a_salvaged_finding_says_it_was_salvaged(tmp_path):
    """Recovery is a guess, and a guess that hides itself cannot be audited."""
    got = parse_contrast_finding(_traj_with(tmp_path, REAL_CORRUPT_BLOCK), "t")
    assert got["parse_status"] == PARSE_SALVAGED


def test_a_clean_finding_is_marked_parsed_not_salvaged(tmp_path):
    block = json.dumps({"task": "t", "is_culprit": True, "gap": "no retry"})
    got = parse_contrast_finding(_traj_with(tmp_path, block), "t")
    assert got["parse_status"] == PARSE_OK


def test_a_genuine_innocent_verdict_is_marked_parsed(tmp_path):
    """The whole point: this must be distinguishable from an unreadable block."""
    block = json.dumps({"task": "t", "is_culprit": False, "note": "model luck"})
    got = parse_contrast_finding(_traj_with(tmp_path, block), "t")
    assert got["is_culprit"] is False
    assert got["parse_status"] == PARSE_OK


def test_an_unreadable_block_is_not_recorded_as_an_innocent_verdict(tmp_path):
    got = parse_contrast_finding(_traj_with(tmp_path, "not json at all {{{"), "t")
    assert got["parse_status"] == PARSE_FAILED


def test_a_missing_finding_is_not_recorded_as_an_innocent_verdict(tmp_path):
    """No block emitted at all is also "no verdict", not "not the culprit"."""
    assert parse_contrast_finding(None, "t")["parse_status"] == PARSE_FAILED


def test_a_non_object_block_is_not_recorded_as_an_innocent_verdict(tmp_path):
    got = parse_contrast_finding(_traj_with(tmp_path, '["a", "b"]'), "t")
    assert got["parse_status"] == PARSE_FAILED


# --------------------------------------------------------------------------
# Salvage must not invent the verdict it exists to rescue.
#
# The key/value split was a regex with no idea whether it was inside a string
# or inside a nested object. An investigator quoting JSON in its own prose —
# which is exactly what a *broken* block looks like — could therefore hand the
# salvager a second `"is_culprit":` that overwrote the real one. Recovering a
# lost verdict and fabricating one are the same code path; only these tests
# separate them.
# --------------------------------------------------------------------------


# `is_culprit` is true at the top, and the prose in `note` quotes a fragment
# shaped exactly like a later field. The unescaped quotes are what break strict
# JSON in the first place, so this shape and the salvage path always co-occur.
FAKE_KEY_IN_STRING = (
    '{"is_culprit": true, "note": "flag is ,"is_culprit": false, "fixable_now": true}'
)


def test_the_fake_key_fixture_is_genuinely_invalid_json():
    """Guards the fixture: if this parses, salvage never runs and we prove nothing."""
    with pytest.raises(ValueError):
        json.loads(FAKE_KEY_IN_STRING)


def test_a_fake_verdict_inside_a_string_cannot_overwrite_the_real_one(tmp_path):
    got = parse_contrast_finding(_traj_with(tmp_path, FAKE_KEY_IN_STRING), "t")
    assert got["is_culprit"] is True


def test_a_broken_string_costs_later_fields_but_never_the_verdict(tmp_path):
    """The trade this fix makes, stated out loud.

    An impostor key carries an unbalanced `"`, so from that point on nothing
    can tell string from structure and the real `fixable_now` after it is lost
    — the boundary-regex this replaced *did* recover it. That is the price of
    refusing to hand a fabricated verdict to the portfolio, and it is cheap:
    the fields salvage exists for (`is_culprit`, `locked_module`) sit at the
    top of the block, above any prose long enough to break.
    """
    got = parse_contrast_finding(_traj_with(tmp_path, FAKE_KEY_IN_STRING), "t")
    assert got["is_culprit"] is True
    assert "fixable_now" not in got


def test_a_key_nested_inside_an_array_is_not_salvaged_as_a_top_level_field():
    """Lens findings carry `evidence: [{"task": ...}]`; that task is not the block's."""
    raw = '{"is_culprit": true, "evidence": [{"task": "inner"}], "note": "a "b"}'
    got, status = loads_finding_block(raw)
    assert status == PARSE_SALVAGED
    assert "task" not in got


def test_a_salvaged_verdict_that_is_not_a_boolean_is_no_verdict_at_all(tmp_path):
    """`bool()` on a recovered fragment makes every non-empty string a culprit."""
    block = '{"task": "t", "is_culprit": tru, "note": "x "y"}'
    got = parse_contrast_finding(_traj_with(tmp_path, block), "t")
    assert got["parse_status"] == PARSE_FAILED


# --------------------------------------------------------------------------
# The ledger has to carry the distinction, or it dies at the parser boundary.
# --------------------------------------------------------------------------


def _ingest(tmp_path, findings):
    return _backlog.ingest_step(tmp_path, {"step": "s1", "findings": findings})


def test_the_ledger_records_that_a_finding_was_unreadable(tmp_path):
    (rec,) = _ingest(tmp_path, [{"task": "t", "parse_status": PARSE_FAILED}])
    assert rec.parse_status == PARSE_FAILED


def test_an_unreadable_finding_survives_a_write_read_round_trip(tmp_path):
    _ingest(tmp_path, [{"task": "t", "parse_status": PARSE_FAILED}])
    (back,) = _backlog.load_findings(tmp_path)
    assert back.parse_status == PARSE_FAILED


def test_a_finding_with_no_marker_is_treated_as_parsed(tmp_path):
    """Ledgers written before this fix must not all read back as unreadable."""
    (rec,) = _ingest(tmp_path, [{"task": "t", "is_culprit": True}])
    assert rec.parse_status == PARSE_OK


def test_a_ledger_row_written_before_this_field_existed_reads_back_as_parsed(
    tmp_path,
):
    """The on-disk path, which `_record_of`'s own default does NOT cover.

    Every `findings.jsonl` already on disk lacks the column entirely. Defaulting
    it to "failed" would retroactively mark the whole corpus unreadable — the
    exact inverse of the bug this fix exists for, and far louder.
    """
    row = {  # the real pre-fix schema, verbatim
        "finding_id": "f_old",
        "step": "s0",
        "index": 0,
        "task": "t",
        "lens": "tools",
        "is_culprit": True,
        "support_tasks": ["t"],
        "provenance": {},
        "raw_finding": {},
    }
    path = _backlog.findings_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n")

    (back,) = _backlog.load_findings(tmp_path)
    assert back.parse_status == PARSE_OK
