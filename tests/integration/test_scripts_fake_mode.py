"""CI-safe smoke tests for the demo/eval scripts' --fake mode (no network,
no model download) — see docs/design's Phase 3 verification section.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_demo_conversation_fake_mode_exits_cleanly(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/demo_conversation.py", "--fake"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "ADD: Name is Bibek" in result.stdout
    assert "UPDATE:" in result.stdout
    assert "DELETE:" in result.stdout


def test_eval_recall_fake_mode_exits_cleanly_with_zero_leakage():
    result = subprocess.run(
        [sys.executable, "scripts/eval_recall.py", "--fake"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "Cross-user leakage: 0" in result.stdout
    assert "PASSED" in result.stdout
