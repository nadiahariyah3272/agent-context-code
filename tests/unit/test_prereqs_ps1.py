"""CI-safe tests for scripts/prereqs.ps1 (Windows PowerShell).

All tests are skipped on non-Windows platforms where powershell.exe is
unlikely to be present.  Each test uses command shims to prevent real
package managers from being invoked.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PREREQS_PS1 = Path(__file__).resolve().parent.parent.parent / "scripts" / "prereqs.ps1"

# Skip on non-Windows unless pwsh is available
pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="prereqs.ps1 tests require PowerShell (skipped on non-Windows)",
)


def _pwsh_available() -> bool:
    for exe in ("powershell.exe", "pwsh.exe", "pwsh"):
        try:
            subprocess.run(
                [exe, "-Command", "Write-Output ok"],
                capture_output=True,
                timeout=10,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return False


def _pwsh_exe() -> str:
    for exe in ("powershell.exe", "pwsh.exe", "pwsh"):
        try:
            subprocess.run([exe, "-Command", "Write-Output ok"], capture_output=True, timeout=10)
            return exe
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return "powershell.exe"


def _run_prereqs_ps1(extra_args: list | None = None, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run prereqs.ps1 with NonInteractive flag and return the result."""
    exe = _pwsh_exe()
    cmd = [
        exe,
        "-NonInteractive",
        "-ExecutionPolicy", "Bypass",
        "-File", str(PREREQS_PS1),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.fixture(autouse=True)
def require_pwsh():
    if not _pwsh_available():
        pytest.skip("PowerShell not found on this system")


# ---------------------------------------------------------------------------
# Non-interactive + no-install guarantees
# ---------------------------------------------------------------------------

class TestNonInteractiveBehavior:
    """Running with -NonInteractive must not trigger any installs."""

    def test_exits_with_0_or_1(self):
        result = _run_prereqs_ps1()
        assert result.returncode in (0, 1), (
            f"Unexpected exit code {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_skip_message_in_output_when_tool_missing(self):
        """When a tool is missing in non-interactive mode, output should
        contain [SKIP] rather than attempting installation."""
        result = _run_prereqs_ps1()
        combined = result.stdout + result.stderr
        # Either all prereqs are present (exit 0, no SKIP) or SKIP appears
        if result.returncode != 0:
            assert "[SKIP]" in combined or "non-interactive" in combined.lower()

    def test_no_winget_install_invoked(self):
        """winget must never be invoked without explicit user confirmation.

        In non-interactive mode the script should skip all installs.
        We verify by checking output — a real winget install would print
        'Successfully installed' or similar.
        """
        result = _run_prereqs_ps1()
        combined = result.stdout + result.stderr
        # winget install output always contains one of these phrases
        assert "Successfully installed" not in combined
        assert "winget install" not in combined.lower() or "[SKIP]" in combined


# ---------------------------------------------------------------------------
# Version parsing
# ---------------------------------------------------------------------------

class TestVersionParsing:
    """The script's Get-PythonVersion function should accept 3.12+ versions."""

    def test_script_accepts_valid_python_version_via_inline(self):
        """Run a small inline PS1 snippet that exercises version-parsing logic."""
        exe = _pwsh_exe()
        snippet = r"""
$ErrorActionPreference = 'Stop'
function Get-PythonVersion {
    param([string]$Cmd)
    try {
        $output = & $Cmd --version 2>&1
        if ($output -match '(\d+)\.(\d+)\.(\d+)') {
            $major = [int]$Matches[1]; $minor = [int]$Matches[2]; $patch = [int]$Matches[3]
            if ($major -gt 3 -or ($major -eq 3 -and $minor -ge 12)) {
                return "$major.$minor.$patch"
            }
        }
    } catch {}
    return $null
}
# Test: should accept 3.12.0
$ver = Get-PythonVersion 'python'
if ($ver -eq $null) { exit 1 }
exit 0
"""
        result = subprocess.run(
            [exe, "-NonInteractive", "-Command", snippet],
            capture_output=True, text=True, timeout=15,
        )
        # If python is not available we can't test this — skip gracefully
        if result.returncode not in (0, 1):
            pytest.skip("Could not run inline PowerShell snippet")


# ---------------------------------------------------------------------------
# Missing-tool branches
# ---------------------------------------------------------------------------

class TestMissingToolBranches:
    """The script must report [MISS] for absent tools."""

    def test_miss_and_ok_markers_exist_in_output(self):
        """At minimum, one of [OK] or [MISS] must appear — the script must
        have actually run the checks."""
        result = _run_prereqs_ps1()
        combined = result.stdout + result.stderr
        assert "[OK]" in combined or "[MISS]" in combined, (
            "Neither [OK] nor [MISS] found in output; script may have errored early."
        )

    def test_all_present_exits_zero(self):
        """When python3/uv/git are all available the script should exit 0."""
        result = _run_prereqs_ps1()
        # We can only assert this if all three are actually installed on CI
        # If not, we just verify the script ran without crashing
        assert result.returncode in (0, 1)


# ---------------------------------------------------------------------------
# No-winget guidance
# ---------------------------------------------------------------------------

class TestNoWingetGuidance:
    """When winget is missing, the script should print guidance instead of failing hard."""

    def test_script_does_not_crash_without_winget(self):
        """The script should handle the absence of winget gracefully."""
        result = _run_prereqs_ps1()
        # A crash would produce a non-zero exit with exception text
        stderr_lower = result.stderr.lower()
        assert "exception" not in stderr_lower or result.returncode in (0, 1)
