# PyPI Publishing Guide — AGENT Context Local

A personal walkthrough for publishing `agent-context-local` to PyPI. Steps are
ordered as you'd actually do them for a release, with project-specific gotchas
called out inline.

---

## Table of Contents

- [One-Time Account Setup](#one-time-account-setup)
- [Pre-Flight Checklist](#pre-flight-checklist)
- [Build the Package](#build-the-package)
- [Smoke Test the Build Locally](#smoke-test-the-build-locally)
- [Publish to TestPyPI First](#publish-to-testpypi-first)
- [Validate on TestPyPI](#validate-on-testpypi)
- [Publish to Production PyPI](#publish-to-production-pypi)
- [Post-Release Steps](#post-release-steps)
- [Platform Test Matrix](#platform-test-matrix)
- [Torch / GPU Extras Caveat](#torch--gpu-extras-caveat)
- [Automating Releases with GitHub Actions](#automating-releases-with-github-actions)

---

## One-Time Account Setup

You only need to do these steps once.

### 1. Create accounts

- **TestPyPI:** https://test.pypi.org/account/register/
- **PyPI:** https://pypi.org/account/register/

Use the same email for both. Enable 2FA on both accounts (PyPI now requires it
for publishing).

### 2. Generate API tokens

Generate a scoped token for each registry:

1. Log in to TestPyPI → Account settings → API tokens → **Add API token**
   - Token name: `agent-context-local-testpypi`
   - Scope: **Entire account** (until the project exists; scope it to the
     project on subsequent releases)
2. Repeat on PyPI for `agent-context-local-pypi`

Store both tokens somewhere safe (password manager). You will not see them again.

### 3. Configure token credentials

The easiest approach is a `~/.pypirc` file:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-<your-real-pypi-token>

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-<your-real-testpypi-token>
```

**Windows path:** `C:\Users\<you>\.pypirc`

Restrict file permissions on Linux/macOS:

```bash
chmod 600 ~/.pypirc
```

---

## Pre-Flight Checklist

Run through these before every release.

### Version bump

The current version is set in `pyproject.toml`:

```toml
[project]
name = "agent-context-local"
version = "0.9.0"
```

Decide your next version (follow [SemVer](https://semver.org/)):
- Patch bump (`0.9.0` → `0.9.1`): bug fixes only
- Minor bump (`0.9.0` → `0.10.0`): new features, backwards-compatible
- Major bump (`0.9.0` → `1.0.0`): breaking changes

Update `pyproject.toml` and commit the bump before building.

### Verify metadata is clean

```bash
# Check the package name is correct
grep '^name' pyproject.toml
# Expected: name = "agent-context-local"

# Check entry points resolve
grep -A5 '\[project.scripts\]' pyproject.toml
# Expected:
#   agent-context-local     = "scripts.cli:main"
#   agent-context-local-mcp = "mcp_server.server:main"
```

### Run the test suite

```bash
uv run python tests/run_tests.py
uv run python -m pytest tests/unit/test_cli.py -v
uv run python -m pytest tests/test_lancedb_schema.py -v
```

All tests must pass before publishing.

### Clean the dist directory

```bash
rm -rf dist/
```

---

## Build the Package

```bash
uv build
```

This produces two artifacts in `dist/`:

| File | Purpose |
|------|---------|
| `agent_context_local-<version>-py3-none-any.whl` | Wheel (preferred by installers) |
| `agent_context_local-<version>.tar.gz` | Source distribution (sdist) |

Both should be uploaded to PyPI together.

### Inspect the wheel contents

Verify the wheel includes everything it should:

```bash
# List files in the wheel (it's just a zip)
python -c "import zipfile, sys; [print(n) for n in zipfile.ZipFile('dist/agent_context_local-$(grep '^version' pyproject.toml | cut -d'\"' -f2)-py3-none-any.whl').namelist()]"
```

Look for:
- `mcp_server/strings.yaml` (packaged data file)
- `py.typed` markers in each subpackage
- All subpackages: `chunking/`, `embeddings/`, `search/`, `mcp_server/`,
  `merkle/`, `reranking/`, `graph/`, `scripts/`
- `common_utils.py` (top-level py module)

If anything is missing, check `[tool.setuptools.packages.find]` and
`[tool.setuptools.package-data]` in `pyproject.toml`.

### Validate package metadata

```bash
uvx twine check dist/*
```

This catches malformed metadata (long description rendering errors, missing
required fields, etc.) before upload. Fix any warnings before proceeding.

---

## Smoke Test the Build Locally

Before touching any registry, install the wheel in a fresh isolated environment
and verify the entry points work.

```bash
# Create a throwaway venv
uv venv /tmp/acl-smoke-test
source /tmp/acl-smoke-test/bin/activate    # Linux/macOS
# or: /tmp/acl-smoke-test/Scripts/activate  # Windows

# Install the wheel (not from PyPI — from your local dist/)
pip install dist/agent_context_local-*.whl

# Check the CLI entry point is registered
agent-context-local --help

# Check the MCP server entry point is registered
agent-context-local-mcp --help

# Deactivate
deactivate
rm -rf /tmp/acl-smoke-test
```

> **Note:** `uv tool install` doesn't support installing from a local path
> the same way pip does, so use a plain venv for this smoke test. The important
> thing is verifying the wheel is installable and entry points are wired up.

---

## Publish to TestPyPI First

**Always publish to TestPyPI before production.** It is a separate registry
with the same upload mechanics but zero consequence if something goes wrong.

```bash
uvx twine upload --repository testpypi dist/*
```

You should see output like:

```
Uploading distributions to https://test.pypi.org/legacy/
Uploading agent_context_local-0.9.0-py3-none-any.whl
Uploading agent_context_local-0.9.0.tar.gz
View at: https://test.pypi.org/project/agent-context-local/0.9.0/
```

Visit that URL and verify the project page looks right: description renders
correctly, classifiers are appropriate, links are valid.

---

## Validate on TestPyPI

Install from TestPyPI in a clean environment and run a basic functional test.

```bash
# Install from TestPyPI (dependencies still come from real PyPI)
uv tool install agent-context-local \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/

# Verify
agent-context-local --version
agent-context-local doctor
agent-context-local-mcp --help
```

> **Tip:** TestPyPI and PyPI are separate indexes, so packages that exist only
> on real PyPI (like `lancedb`, `fastmcp`, etc.) won't be found if you only
> point at TestPyPI. The `--extra-index-url` flag is what lets dependencies
> resolve from the real PyPI while pulling your package from TestPyPI.

If anything breaks here, fix it and re-upload to TestPyPI. You can upload
the same version multiple times to TestPyPI by deleting the old release from
the TestPyPI web UI first (or by bumping the version).

---

## Publish to Production PyPI

Once TestPyPI validation passes:

```bash
uvx twine upload dist/*
```

That's it. Visit https://pypi.org/project/agent-context-local/ to confirm
the release is live.

### Verify the live package installs

```bash
# Wait ~60 seconds for PyPI CDN propagation, then:
uv tool install agent-context-local

agent-context-local --version
agent-context-local doctor
```

---

## Post-Release Steps

- **Tag the release in git:**

  ```bash
  git tag v0.9.0
  git push origin v0.9.0
  ```

- **Create a GitHub release** from the tag (include changelog notes).

- **Update `MEMORY.md`** if the published version changes anything about the
  install story.

- **Run the platform test matrix** (see below).

---

## Platform Test Matrix

After each release, validate the install on every platform you have access to.

| Platform | Machine | Status | Notes |
|----------|---------|--------|-------|
| Windows 11 | Dev machine | Verified during development | Already working |
| Linux (AMD ROCm) | **Strix Halo machine** | **Must test each release** | See below |
| macOS | No test machine | Untested — assume OK if Windows + Linux pass | |

### Linux validation — AMD Strix Halo machine

> **Reminder:** After every PyPI release, SSH into the Strix Halo Linux box
> and run the full install + functional test. This is currently the only Linux
> machine available for validation. If it passes on Windows and on Strix Halo,
> macOS should follow (same Unix path handling, same dependency graph — no
> reason to expect divergence given the `OS Independent` classifier and no
> native extensions).

On the Strix Halo machine:

```bash
# Fresh install from PyPI
uv tool install agent-context-local

# Basic sanity
agent-context-local --version
agent-context-local doctor

# ROCm-accelerated install (if testing GPU path)
# The ROCm extra installs the ROCm PyTorch build — see the torch caveat below
uv tool install agent-context-local --extra rocm \
  --extra-index-url https://download.pytorch.org/whl/rocm7.1

# Confirm the MCP server starts
agent-context-local-mcp &
sleep 5
kill %1
```

Check that:
- [ ] Install completes without errors
- [ ] `doctor` reports the embedding model as available (or downloads it)
- [ ] MCP server starts and exits cleanly
- [ ] No path-separator or file-permission errors in output

### macOS note

No Mac hardware is currently available for testing. The package has
`Operating System :: OS Independent` in its classifiers, uses no native
C extensions, and all tree-sitter grammars ship as pure-Python wheels. If
Windows and Linux both pass, macOS should work. File issues if users report
otherwise.

---

## Torch / GPU Extras Caveat

This project has a significant packaging nuance around PyTorch and GPU support.

### How it works in development (uv)

In the source checkout, `[tool.uv.sources]` in `pyproject.toml` routes the
`torch` dependency to the correct PyTorch index based on the active extra:

```toml
[tool.uv.sources]
torch = [
    { index = "pytorch-cu126", extra = "cu126" },
    { index = "pytorch-cu128", extra = "cu128" },
    { index = "pytorch-rocm",  extra = "rocm"  },
]
```

This is a **uv-specific feature**. It is not honored by pip or PyPI's resolver.

### What happens on PyPI

When a user installs via `uv tool install agent-context-local`:

- **No extra** (`uv tool install agent-context-local`):
  `torch>=2.10.0` resolves to the standard CPU wheel from PyPI. This is
  correct and expected for CPU-only users.

- **GPU extra** (`uv tool install agent-context-local --extra cu128`):
  The `[tool.uv.sources]` routing applies **if the user is installing via
  uv**. But if they're using pip, the extra resolves to the same
  `torch>=2.10.0` constraint (no index routing) — they'd get CPU torch unless
  they manually pass the PyTorch index URL.

### What to tell users

For GPU users installing from PyPI:

```bash
# CUDA 12.8
uv tool install agent-context-local --extra cu128 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple/

# ROCm 7.1 (Linux only)
uv tool install agent-context-local --extra rocm \
  --index-url https://download.pytorch.org/whl/rocm7.1 \
  --extra-index-url https://pypi.org/simple/
```

The install script (`scripts/install.sh` / `scripts/install.ps1`) handles this
automatically — it's the recommended GPU install path for non-technical users.

---

## Automating Releases with GitHub Actions

Once the manual process is solid, a GitHub Actions workflow can automate it.

### Recommended approach: trusted publishers (no long-lived tokens)

PyPI supports OIDC-based trusted publishers. The workflow authenticates via
GitHub's OIDC token — no secret API tokens needed in your repo.

**Setup (one time per registry):**

1. Go to https://pypi.org/manage/project/agent-context-local/settings/publishing/
2. Add a new trusted publisher:
   - Owner: `tlines2016`
   - Repository: `agent-context-code`
   - Workflow name: `publish.yml`
   - Environment name: `pypi` (optional but recommended)
3. Repeat on TestPyPI.

**Workflow file** (`.github/workflows/publish.yml`):

```yaml
name: Publish to PyPI

on:
  push:
    tags:
      - 'v*'

permissions:
  contents: read
  id-token: write   # required for trusted publisher OIDC

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv build
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish-testpypi:
    needs: build
    runs-on: ubuntu-latest
    environment: testpypi
    steps:
      - uses: actions/download-artifact@v4
        with: { name: dist, path: dist/ }
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/

  publish-pypi:
    needs: publish-testpypi
    runs-on: ubuntu-latest
    environment: pypi
    steps:
      - uses: actions/download-artifact@v4
        with: { name: dist, path: dist/ }
      - uses: pypa/gh-action-pypi-publish@release/v1
```

Push a tag to trigger the workflow:

```bash
git tag v0.9.0
git push origin v0.9.0
```

The workflow builds → publishes to TestPyPI → publishes to production PyPI.
Add a test step between build and publish-testpypi to gate on passing tests.
