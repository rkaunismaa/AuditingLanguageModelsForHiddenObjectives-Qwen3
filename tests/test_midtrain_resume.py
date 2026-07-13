import os

from src.train.midtrain import resolve_resume_checkpoint


def _make_checkpoint(root, step, complete=True):
    """Create a checkpoint-<step> dir; `complete` writes the trainer_state.json
    that HF emits at the end of a save (our completeness marker)."""
    d = root / f"checkpoint-{step}"
    d.mkdir()
    if complete:
        (d / "trainer_state.json").write_text("{}")
    return d


def test_fresh_returns_none_even_with_checkpoints(tmp_path):
    """--fresh forces a clean start regardless of existing checkpoints."""
    _make_checkpoint(tmp_path, 10)
    assert resolve_resume_checkpoint(str(tmp_path), fresh=True) is None


def test_missing_dir_returns_none(tmp_path):
    """No checkpoint dir yet (first run) -> start fresh."""
    assert resolve_resume_checkpoint(str(tmp_path / "nope"), fresh=False) is None


def test_empty_dir_returns_none(tmp_path):
    """Dir exists but holds no checkpoint-* subdir -> nothing to resume."""
    assert resolve_resume_checkpoint(str(tmp_path), fresh=False) is None


def test_picks_highest_complete_checkpoint(tmp_path):
    """Resume from the latest (highest-step) complete checkpoint."""
    for n in (2, 10, 4):
        _make_checkpoint(tmp_path, n)
    got = resolve_resume_checkpoint(str(tmp_path), fresh=False)
    assert got is not None
    assert os.path.basename(got) == "checkpoint-10"


def test_skips_incomplete_latest_checkpoint(tmp_path):
    """A half-written latest checkpoint (no trainer_state.json, e.g. an
    interrupt mid-save) is skipped for the newest COMPLETE one."""
    _make_checkpoint(tmp_path, 4, complete=True)
    _make_checkpoint(tmp_path, 6, complete=False)  # interrupted mid-save
    got = resolve_resume_checkpoint(str(tmp_path), fresh=False)
    assert got is not None
    assert os.path.basename(got) == "checkpoint-4"


def test_all_incomplete_returns_none(tmp_path):
    """If the only checkpoint is incomplete, fall back to a fresh start rather
    than crash-loading a corrupt checkpoint."""
    _make_checkpoint(tmp_path, 2, complete=False)
    assert resolve_resume_checkpoint(str(tmp_path), fresh=False) is None
