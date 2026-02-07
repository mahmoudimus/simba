"""Installation routines for registering Neuron with Claude Code."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import simba.neuron.templates


def install_agents(force: bool = False, update_sections: bool = True) -> None:
    """Install or update agent definition files in .claude/agents/."""
    agents_dir = Path(".claude/agents")
    agents_dir.mkdir(parents=True, exist_ok=True)

    print(f"Agent definitions in {agents_dir}...")

    if update_sections:
        for agent_file in agents_dir.glob("*.md"):
            try:
                original = agent_file.read_text()
                updated = simba.neuron.templates.update_managed_sections(original)
                if original != updated:
                    agent_file.write_text(updated)
                    print(f"   {agent_file.name} (updated managed sections)")
            except Exception as exc:
                print(f"   {agent_file.name}: {exc}", file=sys.stderr)

    for filename, content in simba.neuron.templates.AGENT_TEMPLATES.items():
        file_path = agents_dir / filename

        if file_path.exists() and not force:
            continue

        try:
            final_content = simba.neuron.templates.update_managed_sections(
                content.lstrip()
            )
            file_path.write_text(final_content)
            print(f"   {filename} (created)")
        except Exception as exc:
            print(f"   {filename}: {exc}", file=sys.stderr)


def install_routine(
    server_name: str,
    force_agents: bool,
    use_proxy: bool = False,
) -> None:
    """Register the MCP server with Claude CLI and bootstrap agents."""
    install_agents(force_agents)

    if not shutil.which("claude"):
        print(
            "Error: 'claude' executable not found in PATH.",
            file=sys.stderr,
        )
        print("Please install Claude Code first.", file=sys.stderr)
        sys.exit(1)

    project_root = Path.cwd().resolve()
    subcommand = "proxy" if use_proxy else "run"
    server_launch_cmd = [
        sys.executable,
        "-m",
        "simba.neuron",
        subcommand,
        "--root-dir",
        str(project_root),
    ]

    mode_str = "proxy (hot-reload)" if use_proxy else "direct"
    print(f"\nInstalling MCP Server '{server_name}' ({mode_str})...")

    claude_cmd = [
        "claude",
        "mcp",
        "add",
        server_name,
        "--",
        *server_launch_cmd,
    ]

    try:
        subprocess.run(claude_cmd, check=True)
        print(f"\nSuccessfully installed '{server_name}'!")
        print(
            "   You can now use tools like 'truth_query' and "
            "'verify_z3' in this project."
        )
        if use_proxy:
            print("\n   Hot-reload enabled!")
            print("      After editing, run: pkill -HUP -f 'simba.neuron proxy'")
    except subprocess.CalledProcessError as exc:
        print(
            f"\nInstallation failed with code {exc.returncode}",
            file=sys.stderr,
        )
        sys.exit(exc.returncode)
