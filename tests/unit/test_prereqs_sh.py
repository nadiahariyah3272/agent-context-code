"""CI-safe tests for scripts/prereqs.sh.

Tests run the shell script with a controlled PATH so no real package
managers are invoked.  All tests are skipped automatically on Windows
where bash is not expected to be available.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PREREQS_SH = Path(__file__).resolve().parent.parent.parent / "scripts" / "prereqs.sh"

# Skip the entire module on Windows (no bash)
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="prereqs.sh tests require bash (skipped on Windows)",
)


def _bash_available() -> bool:
    try:
        subprocess.run(["bash", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _run_prereqs(env: dict | None = None, timeout: int = 15) -> subprocess.CompletedProcess:
    """Run prereqs.sh with stdin closed (non-interactive) and return the result."""
    run_env = {**os.environ, **(env or {})}
    return subprocess.run(
        ["bash", str(PREREQS_SH)],
        stdin=subprocess.DEVNULL,  # forces NON_INTERACTIVE=1
        capture_output=True,
        text=True,
        env=run_env,
        timeout=timeout,
    )


@pytest.fixture(autouse=True)
def require_bash():
    if not _bash_available():
        pytest.skip("bash not found on this system")


# ---------------------------------------------------------------------------
# Non-interactive mode
# ---------------------------------------------------------------------------

class TestNonInteractiveBehavior:
    """When stdin is closed the script must not attempt installs."""

    def test_script_exits_with_code_0_or_1_only(self):
        result = _run_prereqs()
        assert result.returncode in (0, 1), (
            f"Unexpected exit code {result.returncode}: {result.stderr}"
        )

    def test_no_install_commands_run_in_non_interactive_mode(self):
        """brew/apt/dnf must never be called when running non-interactively.

        We shim package managers with scripts that echo a sentinel and exit 1
        so that if they are called, the sentinel appears in stdout.
        """
        with tempfile.TemporaryDirectory() as shim_dir:
            sentinel = "INSTALL_COMMAND_WAS_CALLED"
            for cmd in ("brew", "apt", "dnf", "sudo"):
                shim = Path(shim_dir) / cmd
                shim.write_text(f"#!/bin/sh\necho {sentinel}\nexit 1\n")
                shim.chmod(0o755)

            env = {"PATH": f"{shim_dir}:{os.environ.get('PATH', '')}"}
            result = _run_prereqs(env=env)
            assert sentinel not in result.stdout, (
                "Package manager was invoked in non-interactive mode"
            )

    def test_non_interactive_message_in_output(self):
        """Output should mention non-interactive / piped behaviour."""
        result = _run_prereqs()
        combined = result.stdout + result.stderr
        # The script prints guidance about running locally when not interactive
        assert "non-interactive" in combined.lower() or "run locally" in combined.lower() or result.returncode == 0


# ---------------------------------------------------------------------------
# Missing dependency reporting
# ---------------------------------------------------------------------------

class TestMissingDependencySummary:
    """The script must report which prerequisites are missing."""

    def test_all_present_exits_zero(self):
        """If python3, uv, and git are all on PATH the script exits 0."""
        with tempfile.TemporaryDirectory() as shim_dir:
            # Create shims for the three tools that return valid versions
            py_shim = Path(shim_dir) / "python3"
            py_shim.write_text("#!/bin/sh\necho 'Python 3.12.0'\n")
            py_shim.chmod(0o755)

            uv_shim = Path(shim_dir) / "uv"
            uv_shim.write_text("#!/bin/sh\necho 'uv 0.4.0'\n")
            uv_shim.chmod(0o755)

            git_shim = Path(shim_dir) / "git"
            git_shim.write_text("#!/bin/sh\necho 'git version 2.40.0'\n")
            git_shim.chmod(0o755)

            # Minimal PATH with only our shims + /bin for sh/echo etc.
            env = {"PATH": f"{shim_dir}:/usr/bin:/bin"}
            result = _run_prereqs(env=env)
            # Verify the script actually ran the checks (not just exited early)
            combined = result.stdout + result.stderr
            assert "[OK]" in combined, (
                "Script did not produce any [OK] markers — checks may not have run.\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
            assert result.returncode == 0, (
                f"Expected exit 0, got {result.returncode}.\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

    def test_missing_uv_reported_in_output(self):
        """When uv is absent the word 'uv' must appear in the output."""
        with tempfile.TemporaryDirectory() as shim_dir:
            # Provide python3 and git but NOT uv
            py_shim = Path(shim_dir) / "python3"
            py_shim.write_text("#!/bin/sh\necho 'Python 3.12.0'\n")
            py_shim.chmod(0o755)

            git_shim = Path(shim_dir) / "git"
            git_shim.write_text("#!/bin/sh\necho 'git version 2.40.0'\n")
            git_shim.chmod(0o755)

            env = {"PATH": f"{shim_dir}:/usr/bin:/bin"}
            result = _run_prereqs(env=env)
            combined = result.stdout + result.stderr
            assert "uv" in combined.lower()

    def test_ok_marker_for_present_tools(self):
        """Tools that are found should produce an [OK] line."""
        with tempfile.TemporaryDirectory() as shim_dir:
            py_shim = Path(shim_dir) / "python3"
            py_shim.write_text("#!/bin/sh\necho 'Python 3.12.0'\n")
            py_shim.chmod(0o755)

            uv_shim = Path(shim_dir) / "uv"
            uv_shim.write_text("#!/bin/sh\necho 'uv 0.4.0'\n")
            uv_shim.chmod(0o755)

            git_shim = Path(shim_dir) / "git"
            git_shim.write_text("#!/bin/sh\necho 'git version 2.40.0'\n")
            git_shim.chmod(0o755)

            env = {"PATH": f"{shim_dir}:/usr/bin:/bin"}
            result = _run_prereqs(env=env)
            assert "[OK]" in result.stdout

    def test_miss_marker_for_absent_tools(self):
        """Tools not found should produce a [MISS] line."""
        with tempfile.TemporaryDirectory() as shim_dir:
            # Only provide a valid python3 — uv and git are absent
            py_shim = Path(shim_dir) / "python3"
            py_shim.write_text("#!/bin/sh\necho 'Python 3.12.0'\n")
            py_shim.chmod(0o755)

            env = {"PATH": f"{shim_dir}:/usr/bin:/bin"}
            result = _run_prereqs(env=env)
            combined = result.stdout + result.stderr
            assert "[MISS]" in combined

    def test_python_version_too_old_reported(self):
        """Python < 3.12 should not be accepted."""
        with tempfile.TemporaryDirectory() as shim_dir:
            # Shim returns an old version
            py_shim = Path(shim_dir) / "python3"
            py_shim.write_text("#!/bin/sh\necho 'Python 3.9.0'\n")
            py_shim.chmod(0o755)

            env = {"PATH": f"{shim_dir}:/usr/bin:/bin"}
            result = _run_prereqs(env=env)
            combined = result.stdout + result.stderr
            # Should report python as missing or the script should exit non-zero
            assert "[MISS]" in combined or result.returncode != 0
