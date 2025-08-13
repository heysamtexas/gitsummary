"""Command-line interface for git-summary."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import typer
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

from git_summary.config import Config
from git_summary.fetchers import EventCoordinator
from git_summary.github_client import GitHubClient
from git_summary.processors import EventProcessor

app = typer.Typer(
    name="git-summary",
    help="A comprehensive tool for querying the GitHub API to extract, analyze, and summarize user activity",
    no_args_is_help=True,
)

console = Console()


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
    max_events: int | None = typer.Option(
        None,
        "--max-events",
        help="Maximum number of events to process (useful for high-activity users)",
    ),
    # Placeholder filtering options (MVP - not implemented yet)
    include_events: str | None = typer.Option(
        None,
        "--include-events",
        help="Comma-separated list of event types to include (e.g., 'PushEvent,ReleaseEvent')",
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
    if exclude_repos:
        print(
            "[yellow]⚠️  --exclude-repos is not implemented in this MVP version[/yellow]"
        )

    print(f"[green]Analyzing GitHub activity for user: {user}[/green]")
    print(f"[blue]Time range: Last {days} days[/blue]")
    if max_events:
        print(f"[yellow]Event limit: {max_events} events maximum[/yellow]")
    if include_events:
        event_list = include_events.split(",")
        print(f"[magenta]Event filter: {', '.join(event_list)}[/magenta]")
    if output:
        print(f"[cyan]Output will be saved to: {output}[/cyan]")

    # Run the async analysis
    try:
        asyncio.run(
            _analyze_user_activity(
                user, token, days, output, max_events, include_events
            )
        )
    except KeyboardInterrupt:
        print("\n[red]Operation cancelled by user[/red]")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


async def _analyze_user_activity(
    user: str,
    token: str,
    days: int,
    output: str | None,
    max_events: int | None,
    include_events: str | None,
) -> None:
    """Perform the actual GitHub activity analysis."""
    # Calculate date range
    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=days)

    console.print(
        Panel.fit(
            f"[bold cyan]Analyzing GitHub Activity[/bold cyan]\n\n"
            f"[white]User:[/white] [green]{user}[/green]\n"
            f"[white]Period:[/white] [blue]{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}[/blue]\n"
            f"[white]Duration:[/white] [yellow]{days} days[/yellow]",
            title="📊 GitHub Activity Analysis",
        )
    )

    # Initialize GitHub client and coordinator
    client = GitHubClient(token=token)
    coordinator = EventCoordinator(client)
    processor = EventProcessor()

    try:
        # Fetch events with progress indication
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("•"),
            TextColumn("[cyan]{task.fields[events]}[/cyan] events"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            # Create a task for fetching
            fetch_task = progress.add_task(
                f"Fetching events for [green]{user}[/green]...", events=0
            )

            def update_progress(current: int, _total: int | None) -> None:
                progress.update(fetch_task, events=current)

            # Fetch events by date range
            events = await coordinator.fetch_events_by_date_range(
                username=user,
                start_date=start_date,
                end_date=end_date,
                max_pages=None,  # Fetch all available events
                max_events=max_events,
                progress_callback=update_progress,
            )

            progress.update(
                fetch_task, description=f"✓ Fetched events for [green]{user}[/green]"
            )

        console.print(
            f"\\n[green]✓[/green] Successfully fetched [cyan]{len(events)}[/cyan] events"
        )

        # Filter events by type if specified
        if include_events:
            original_count = len(events)
            event_types = [
                event_type.strip() for event_type in include_events.split(",")
            ]
            events = [event for event in events if event.type in event_types]
            console.print(
                f"[magenta]✓[/magenta] Filtered to [cyan]{len(events)}[/cyan] events "
                f"({original_count - len(events)} excluded)"
            )

        # Process events with progress indication
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False,
        ) as progress:
            process_task = progress.add_task(
                "Processing events and generating analysis..."
            )

            # Process the events
            report = processor.process(events, user, start_date, end_date)

            progress.update(
                process_task, description="✓ Completed event processing and analysis"
            )

        console.print("[green]✓[/green] Analysis complete!\\n")

        # Display results
        _display_activity_report(report)

        # Save to file if requested
        if output:
            _save_report_to_file(report, output)
            console.print(f"\\n[green]✓[/green] Results saved to [cyan]{output}[/cyan]")

    except Exception as e:
        console.print(f"\\n[red]✗[/red] Analysis failed: {e}")
        raise
    finally:
        # Clean up client connection
        await client.close()


def _display_activity_report(report: Any) -> None:
    """Display the activity report in a rich console format."""
    # Summary Panel
    summary_content = (
        f"[white]Total Events:[/white] [cyan]{report.summary.total_events}[/cyan]\\n"
        f"[white]Active Repositories:[/white] [green]{report.summary.repositories_active}[/green]\\n"
        f"[white]Most Active Repository:[/white] [yellow]{report.summary.most_active_repository or 'N/A'}[/yellow]\\n"
        f"[white]Most Common Event:[/white] [blue]{report.summary.most_common_event_type or 'N/A'}[/blue]"
    )

    console.print(
        Panel(summary_content, title="📈 Activity Summary", border_style="green")
    )

    # Event Types Breakdown
    if report.summary.event_breakdown:
        event_table = Table(
            title="📊 Event Types Breakdown", show_header=True, header_style="bold blue"
        )
        event_table.add_column("Event Type", style="cyan")
        event_table.add_column("Count", justify="right", style="green")

        # Sort by count descending
        for event_type, count in sorted(
            report.summary.event_breakdown.items(), key=lambda x: x[1], reverse=True
        ):
            event_table.add_row(event_type, str(count))

        console.print(event_table)

    # Daily Activity (if multiple days)
    if len(report.daily_rollups) > 1:
        daily_table = Table(
            title="📅 Daily Activity", show_header=True, header_style="bold blue"
        )
        daily_table.add_column("Date", style="cyan")
        daily_table.add_column("Events", justify="right", style="green")
        daily_table.add_column("Repositories", justify="right", style="yellow")

        for rollup in report.daily_rollups:
            daily_table.add_row(
                rollup.date, str(rollup.events), str(len(rollup.repositories))
            )

        console.print(daily_table)

    # Top Repositories
    if report.repository_breakdown:
        repo_table = Table(
            title="🏆 Repository Activity", show_header=True, header_style="bold blue"
        )
        repo_table.add_column("Repository", style="cyan")
        repo_table.add_column("Events", justify="right", style="green")
        repo_table.add_column("First Activity", style="dim")

        # Show top 10 repositories by activity
        sorted_repos = sorted(
            report.repository_breakdown.items(), key=lambda x: x[1].events, reverse=True
        )[:10]

        for repo_name, breakdown in sorted_repos:
            first_activity = breakdown.first_activity or "N/A"
            if first_activity and first_activity != "N/A":
                # Format the timestamp nicely
                try:
                    dt = datetime.fromisoformat(first_activity.replace("Z", "+00:00"))
                    first_activity = dt.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    pass  # Keep original if parsing fails

            repo_table.add_row(repo_name, str(breakdown.events), first_activity)

        console.print(repo_table)

    # Detailed Repository Activity
    _display_detailed_repository_activity(report)


def _display_detailed_repository_activity(report: Any) -> None:
    """Display detailed activity breakdown by repository with commit messages and release details."""
    if not report.detailed_events:
        return

    # Group events by repository
    repo_events: dict[str, dict[str, list[Any]]] = {}
    for event in report.detailed_events:
        repo_name = event.repository
        if repo_name not in repo_events:
            repo_events[repo_name] = {"PushEvent": [], "ReleaseEvent": [], "other": []}

        if event.type == "PushEvent":
            repo_events[repo_name]["PushEvent"].append(event)
        elif event.type == "ReleaseEvent":
            repo_events[repo_name]["ReleaseEvent"].append(event)
        else:
            repo_events[repo_name]["other"].append(event)

    # Display top 5 most active repositories with details
    sorted_repos = sorted(
        repo_events.items(),
        key=lambda x: len(x[1]["PushEvent"])
        + len(x[1]["ReleaseEvent"])
        + len(x[1]["other"]),
        reverse=True,
    )[:5]

    for repo_name, events in sorted_repos:
        push_events = events["PushEvent"]
        release_events = events["ReleaseEvent"]
        total_events = len(push_events) + len(release_events) + len(events["other"])

        if total_events == 0:
            continue

        console.print(
            f"\n[bold cyan]📁 {repo_name}[/bold cyan] ([green]{total_events}[/green] events)"
        )

        # Show commits
        if push_events:
            console.print(f"  [yellow]💻 {len(push_events)} commits:[/yellow]")
            for event in push_events[:5]:  # Show first 5 commits
                details = event.details
                commit_count = details.get("commits_count", 1)
                branch = details.get("ref", "unknown").replace("refs/heads/", "")

                # Format timestamp
                try:
                    dt = datetime.fromisoformat(event.created_at.replace("Z", "+00:00"))
                    time_str = dt.strftime("%m/%d %H:%M")
                except ValueError:
                    time_str = event.created_at[:10]

                console.print(
                    f"    [dim]{time_str}[/dim] [{commit_count} commit{'s' if commit_count > 1 else ''}] [blue]→ {branch}[/blue]"
                )

                # Show commit messages
                if "commit_messages" in details:
                    for msg in details["commit_messages"][
                        :2
                    ]:  # Show first 2 commit messages
                        # Truncate long commit messages
                        if len(msg) > 80:
                            msg = msg[:77] + "..."
                        console.print(f"      [green]•[/green] {msg}")

                    if details.get("more_commits", 0) > 0:
                        console.print(
                            f"      [dim]... and {details['more_commits']} more commits[/dim]"
                        )

            if len(push_events) > 5:
                console.print(
                    f"    [dim]... and {len(push_events) - 5} more commits[/dim]"
                )

        # Show releases
        if release_events:
            console.print(f"  [yellow]🚀 {len(release_events)} releases:[/yellow]")
            for event in release_events[:3]:  # Show first 3 releases
                details = event.details
                version = details.get("version", "unknown")
                release_name = details.get("release_name", "")

                # Format timestamp
                try:
                    dt = datetime.fromisoformat(event.created_at.replace("Z", "+00:00"))
                    time_str = dt.strftime("%m/%d %H:%M")
                except ValueError:
                    time_str = event.created_at[:10]

                console.print(
                    f"    [dim]{time_str}[/dim] [magenta]{version}[/magenta]", end=""
                )
                if release_name and release_name != version:
                    console.print(f" - {release_name}")
                else:
                    console.print()

                # Show release notes (first line only)
                if "release_notes" in details:
                    notes = details["release_notes"].split("\n")[0]
                    if len(notes) > 100:
                        notes = notes[:97] + "..."
                    console.print(f"      [dim]{notes}[/dim]")

            if len(release_events) > 3:
                console.print(
                    f"    [dim]... and {len(release_events) - 3} more releases[/dim]"
                )


def _save_report_to_file(report: Any, output_path: str) -> None:
    """Save the activity report to a JSON file."""
    output_file = Path(output_path)

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict for JSON serialization
    report_dict = (
        report.model_dump() if hasattr(report, "model_dump") else report.__dict__
    )

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)


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
