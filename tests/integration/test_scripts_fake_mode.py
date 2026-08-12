"""CI-safe smoke tests for the demo/eval scripts' --fake mode (no network,
no model download) — see docs/design's Phase 3 verification section.
"""

import subprocess
import sys
from pathlib import Path

import pytest

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


def test_record_demo_gif_fake_mode_produces_a_valid_multi_frame_gif(tmp_path):
    pytest.importorskip("PIL")  # scripts/record_demo_gif.py needs the [media] extra

    output_path = tmp_path / "test-demo.gif"
    result = subprocess.run(
        [sys.executable, "scripts/record_demo_gif.py", "--output", str(output_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr

    from PIL import Image

    assert output_path.exists()
    with Image.open(output_path) as image:
        assert image.format == "GIF"
        assert image.n_frames > 5
