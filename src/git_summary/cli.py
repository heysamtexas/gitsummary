"""Command-line interface for git-summary."""

import typer
from rich import print
from rich.prompt import Confirm, Prompt
from rich.table import Table

from git_summary.config import Config

app = typer.Typer(
    name="git-summary",
    help="A comprehensive tool for querying the GitHub API to extract, analyze, and summarize user activity",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Show the version and exit."""
    from git_summary import __version__

    print(f"git-summary {__version__}")


@app.command()
def summary(
    user: str = typer.Argument(None, help="GitHub username to analyze"),
    token: str | None = typer.Option(
        None,
        "--token",
        "-t",
        help="GitHub Personal Access Token (or set GITHUB_TOKEN env var)",
        envvar="GITHUB_TOKEN",
    ),
    days: int = typer.Option(7, "--days", "-d", help="Number of days to analyze"),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file (JSON format). If not specified, prints to stdout",
    ),
    # Placeholder filtering options (MVP - not implemented yet)
    include_events: str | None = typer.Option(
        None,
        "--include-events",
        help="[MVP: Not implemented] Comma-separated list of event types to include",
    ),
    exclude_repos: str | None = typer.Option(
        None,
        "--exclude-repos",
        help="[MVP: Not implemented] Comma-separated list of repositories to exclude",
    ),
) -> None:
    """Generate a summary of GitHub activity for a user."""
    config = Config()

    # Interactive mode: get username if not provided
    if not user:
        user = Prompt.ask("[cyan]Enter GitHub username to analyze")
        if not user:
            print("[red]Error: Username is required[/red]")
            raise typer.Exit(1)

    # Get token from CLI arg, env var, or stored config
    if not token:
        token = config.get_token()

    # Interactive mode: prompt for token if none found
    if not token:
        print("[yellow]No GitHub token found.[/yellow]")
        print(
            "You can get a Personal Access Token from: https://github.com/settings/tokens"
        )

        if Confirm.ask("Would you like to enter a token now?"):
            token = Prompt.ask(
                "[cyan]Enter your GitHub Personal Access Token", password=True
            )

            if token and Confirm.ask("Save this token for future use?"):
                config.set_token(token)

        if not token:
            print("[red]Error: GitHub token is required to continue[/red]")
            raise typer.Exit(1)

    # Show MVP warnings for unimplemented features
    if include_events:
        print(
            "[yellow]⚠️  --include-events is not implemented in this MVP version[/yellow]"
        )
    if exclude_repos:
        print(
            "[yellow]⚠️  --exclude-repos is not implemented in this MVP version[/yellow]"
        )

    print(f"[green]Analyzing GitHub activity for user: {user}[/green]")
    print(f"[blue]Time range: Last {days} days[/blue]")
    if output:
        print(f"[cyan]Output will be saved to: {output}[/cyan]")

    # TODO: Implement actual GitHub API integration
    print("[yellow]Note: Implementation coming soon![/yellow]")


@app.command()
def auth() -> None:
    """Manage GitHub authentication (interactive setup)."""
    config = Config()

    print("[bold cyan]GitHub Authentication Setup[/bold cyan]")
    print()
    print("To use git-summary, you need a GitHub Personal Access Token.")
    print("You can create one at: [link]https://github.com/settings/tokens[/link]")
    print()
    print("[dim]Required scopes: public_repo (or repo for private repos)[/dim]")
    print()

    current_token = config.get_token()
    if current_token:
        print("[green]✓[/green] You already have a token stored locally")

        if not Confirm.ask("Would you like to replace it with a new token?"):
            return

    token = Prompt.ask("[cyan]Enter your GitHub Personal Access Token", password=True)

    if not token:
        print("[red]No token provided[/red]")
        return

    config.set_token(token)
    print("[green]✓[/green] Authentication setup complete!")
    print("You can now use git-summary without specifying a token.")


@app.command()
def auth_status() -> None:
    """Show current authentication status."""
    config = Config()
    info = config.get_config_info()

    table = Table(title="Authentication Status")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Config File", info["config_file"])
    table.add_row("Token Stored", "✓ Yes" if info["has_token"] else "✗ No")

    if info["config_exists"]:
        table.add_row("File Permissions", info["config_file_permissions"] or "unknown")

    print(table)

    if not info["has_token"]:
        print()
        print(
            "[yellow]No token found. Run [bold]git-summary auth[/bold] to set up authentication.[/yellow]"
        )


@app.command()
def auth_remove() -> None:
    """Remove stored authentication token."""
    config = Config()

    if not config.get_token():
        print("[yellow]No token is currently stored[/yellow]")
        return

    if Confirm.ask("[red]Are you sure you want to remove the stored token?[/red]"):
        config.remove_token()
        print("[green]✓[/green] Token removed successfully")
    else:
        print("Token removal cancelled")


if __name__ == "__main__":
    app()
