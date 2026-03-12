"""FastMCP server for AI coding assistant integration - main entry point."""
import logging
import os

try:
    from common_utils import VERSION, is_installed_package, detect_gpu_index_url  # works when installed as package
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from common_utils import VERSION, is_installed_package, detect_gpu_index_url

from mcp_server.code_search_server import CodeSearchServer
from mcp_server.code_search_mcp import CodeSearchMCP


def _ensure_gpu_configured() -> None:
    """One-time GPU auto-detection at server startup.

    If GPU hardware is detected but uv.toml doesn't exist (meaning no
    install script or gpu-setup was run), auto-configure GPU PyTorch.
    Takes effect on next server restart.
    """
    import subprocess
    from pathlib import Path

    project_dir = Path(__file__).resolve().parent.parent
    uv_toml = project_dir / "uv.toml"

    # Skip if already configured (install script or gpu-setup already ran)
    if uv_toml.exists():
        return

    # Skip if running as installed package (no pyproject.toml to work with)
    if not (project_dir / "pyproject.toml").exists():
        return

    # Detect GPU hardware (without relying on torch — CPU torch can't detect GPUs)
    vendor, _ver, _name, index_url = detect_gpu_index_url()

    if not index_url:
        return  # No GPU, MPS (needs no special index), or unsupported

    # Write uv.toml
    uv_toml.write_text(
        f"# Auto-generated — GPU detected at server startup.\n"
        f"# To revert to CPU: delete this file or run gpu-setup --cpu\n\n"
        f"[[index]]\nname = \"pytorch\"\n"
        f"url = \"{index_url}\"\nexplicit = true\n"
    )

    # Re-lock in background (non-blocking so server starts quickly).
    # The next `uv run` (on server restart) will sync automatically.
    subprocess.Popen(
        ["uv", "lock", "--upgrade-package", "torch"],
        cwd=project_dir,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    logger = logging.getLogger(__name__)
    logger.info(
        "GPU detected — configured GPU PyTorch (%s). "
        "GPU acceleration will be active on next server restart.",
        index_url.split("/")[-1],
    )


def _configure_logging(verbose: bool = False) -> None:
    """Set up logging with sensible defaults.

    Default level is WARNING.  Override via ``AGENT_CONTEXT_LOG_LEVEL`` env
    var (DEBUG, INFO, WARNING, ERROR) or the ``--verbose`` / ``-v`` CLI flag.
    """
    env_level = os.environ.get("AGENT_CONTEXT_LOG_LEVEL", "").upper()
    if verbose:
        level = logging.DEBUG
    elif env_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        level = getattr(logging, env_level)
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("mcp").setLevel(level)
    logging.getLogger("fastmcp").setLevel(level)


def main():
    """Main entry point for the server."""
    import argparse

    # Build epilog dynamically based on install mode
    if is_installed_package():
        mcp_cmd = "agent-context-local-mcp"
        register_example = (
            "Register with your MCP client (example for Claude Code):\n"
            "  claude mcp add code-search --scope user -- agent-context-local-mcp\n"
        )
    else:
        mcp_cmd = "uv run --directory <install-dir> python mcp_server/server.py"
        register_example = (
            "Register with your MCP client (example for Claude Code):\n"
            f"  claude mcp add code-search --scope user -- {mcp_cmd}\n"
        )

    parser = argparse.ArgumentParser(
        description="Code Search MCP Server – local semantic code search for AI coding assistants.",
        epilog=(
            "Examples:\n"
            f"  {mcp_cmd}                    # Start with default stdio transport\n"
            f"  {mcp_cmd} --transport sse    # Start with Server-Sent Events transport\n"
            f"  {mcp_cmd} --version          # Show version and exit\n"
            "\n"
            f"{register_example}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"agent-context-local {VERSION}",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http"],
        default="stdio",
        help="Transport protocol to use (default: stdio)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging (overrides AGENT_CONTEXT_LOG_LEVEL)"
    )
    args = parser.parse_args()

    _configure_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    # Auto-detect GPU and write uv.toml if needed (takes effect on next restart)
    try:
        _ensure_gpu_configured()
    except Exception:
        pass  # Non-critical — don't block server startup

    logger.info("Starting Code Search MCP Server v%s (transport=%s)", VERSION, args.transport)

    # Create and run server
    server = CodeSearchServer()
    mcp_server = CodeSearchMCP(server)
    mcp_server.run(transport=args.transport)


if __name__ == "__main__":
    main()
