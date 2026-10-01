"""The tour placement module is JavaScript; pytest runs its node:test suite."""

import shutil
import subprocess
from pathlib import Path

import pytest

SUITE = Path(__file__).parent / "js" / "tour-place.test.mjs"


def test_tour_place_suite_passes() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.fail("Node.js is required to run the tour placement tests")
    # S603: fixed argv (the resolved node binary and a suite path inside this repo), no shell.
    result = subprocess.run(  # noqa: S603
        [node, "--test", str(SUITE)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "# pass 8" in result.stdout, result.stdout
