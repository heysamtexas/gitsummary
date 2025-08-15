"""Command-line interface for git-summary."""

import asyncio
import json
import signal
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from git_summary.ai.context import GitHubEventLike

import typer
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

# Library API imports (clean public interface)
from git_summary import (
    AnalysisConfig,
    AnalysisResult,
    Config,
    ConfigurationError,
    GitHubAnalyzer,
    InvalidTokenError,
    UserNotFoundError,
)

# Temporary imports for legacy functions (to be removed)
from git_summary.adaptive_discovery import AdaptiveRepositoryDiscovery

# Internal imports still needed for AI features and token validation
from git_summary.ai.orchestrator import ActivitySummarizer
from git_summary.ai.personas import PersonaManager
from git_summary.github_client import GitHubClient
from git_summary.processors import EventProcessor

app = typer.Typer(
    name="git-summary",
    help="A comprehensive tool for querying the GitHub API to extract, analyze, and summarize user activity",
    no_args_is_help=True,
)

console = Console()


def _validate_cli_params(
    days: int | None = None,
    max_repos: int | None = None,
    force_strategy: str | None = None,
    max_events: int | None = None,
) -> None:
    """Validate CLI parameters before creating config.

    Args:
        days: Number of days to analyze
        max_repos: Maximum repositories to analyze
        force_strategy: Forced analysis strategy
        max_events: Maximum events to process

    Raises:
        typer.BadParameter: If any parameter is invalid
    """
    errors = []

    if days is not None and days < 1:
        errors.append("--days must be positive")
    if days is not None and days > 90:
        errors.append("--days cannot exceed 90 (GitHub API limitation)")

    if max_repos is not None and max_repos < 1:
        errors.append("--max-repos must be positive")

    if max_events is not None and max_events < 1:
        errors.append("--max-events must be positive")

    if force_strategy and force_strategy not in ["intelligence_guided", "multi_source"]:
        errors.append(
            f"Invalid --force-strategy: {force_strategy}. Must be 'intelligence_guided' or 'multi_source'"
        )

    if errors:
        raise typer.BadParameter("\n".join(errors))


def _print_analysis_summary(result: AnalysisResult) -> None:
    """Print a nicely formatted analysis summary to the console."""
    # Summary panel
    summary_text = (
        f"[bold cyan]Analysis Complete![/bold cyan]\n\n"
        f"[white]User:[/white] [green]{result.username}[/green]\n"
        f"[white]Period:[/white] [blue]{result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}[/blue]\n"
        f"[white]Duration:[/white] [yellow]{result.analysis_period_days} days[/yellow]\n"
        f"[white]Strategy:[/white] [magenta]{result.analysis_strategy}[/magenta]"
    )

    if result.execution_time_ms:
        summary_text += f"\n[white]Execution Time:[/white] [cyan]{result.execution_time_ms / 1000:.1f}s[/cyan]"

    console.print(Panel.fit(summary_text, title="📊 GitHub Activity Summary"))

    # Statistics table
    stats_table = Table(
        title="Activity Statistics", show_header=True, header_style="bold magenta"
    )
    stats_table.add_column("Metric", style="cyan", no_wrap=True)
    stats_table.add_column("Value", style="green")

    stats_table.add_row("Total Events", str(result.total_events))
    stats_table.add_row("Repositories", str(result.total_repositories))
    stats_table.add_row("Commits (Push Events)", str(result.total_commits))
    stats_table.add_row("Pull Requests", str(result.total_pull_requests))
    stats_table.add_row("Issues", str(result.total_issues))

    if result.most_active_repository:
        stats_table.add_row("Most Active Repository", result.most_active_repository)
    if result.most_common_event_type:
        stats_table.add_row("Most Common Event", result.most_common_event_type)

    console.print(stats_table)

    # Event breakdown table
    if result.event_breakdown:
        event_table = Table(
            title="Event Type Breakdown", show_header=True, header_style="bold magenta"
        )
        event_table.add_column("Event Type", style="cyan", no_wrap=True)
        event_table.add_column("Count", style="green", justify="right")

        # Sort by count (descending)
        for event_type, count in sorted(
            result.event_breakdown.items(), key=lambda x: x[1], reverse=True
        ):
            event_table.add_row(event_type, str(count))

        console.print(event_table)

    # Performance metrics
    if hasattr(result, "total_api_calls") and result.total_api_calls > 0:
        perf_text = (
            f"[dim]Performance: {result.total_api_calls} API calls, "
            f"{result.repositories_discovered} repositories discovered[/dim]"
        )
        console.print(perf_text)


async def validate_github_token(token: str) -> bool:
    """Validate GitHub token by making a simple API call.

    Args:
        token: GitHub Personal Access Token

    Returns:
        True if token is valid, False otherwise
    """
    try:
        async with GitHubClient(token=token) as client:
            # Use built-in token validation method - returns dict with validation info
            result = await client.validate_token()
            # Check if validation was successful (presence of user info indicates success)
            return bool(result.get("login"))
    except Exception:
        return False


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
    # Strategy selection options
    comprehensive: bool = typer.Option(
        False,
        "--comprehensive",
        help="Force comprehensive analysis using multi-source discovery. Overcomes 300-event API limit for complete activity coverage. Best for power users and automation accounts.",
    ),
    max_repos: int | None = typer.Option(
        None,
        "--max-repos",
        help="Limit number of repositories analyzed. Adaptive strategy auto-selects optimal defaults (15 for intelligence-guided, 25-50 for comprehensive). Use to optimize performance.",
    ),
    force_strategy: str | None = typer.Option(
        None,
        "--force-strategy",
        help="Override automatic strategy selection. Options: 'intelligence_guided' (fast, top repos) or 'multi_source' (comprehensive, all activity). Mainly for testing.",
    ),
    # Legacy filtering options (MVP - not implemented yet)
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

    # Validate CLI parameters
    _validate_cli_params(days, max_repos, force_strategy)

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

    # Validate strategy options
    if comprehensive and force_strategy:
        print("[red]Error: Cannot use both --comprehensive and --force-strategy[/red]")
        raise typer.Exit(1)

    # Convert comprehensive flag to strategy override
    strategy_override = None
    if comprehensive:
        strategy_override = "multi_source"
    elif force_strategy:
        if force_strategy not in ["intelligence_guided", "multi_source"]:
            print(
                f"[red]Error: Invalid strategy '{force_strategy}'. Must be 'intelligence_guided' or 'multi_source'[/red]"
            )
            raise typer.Exit(1)
        strategy_override = force_strategy

    # Use library API for analysis
    try:
        # Map CLI parameters to library configuration
        config_builder = AnalysisConfig.builder().days(days)

        if max_events:
            config_builder = config_builder.max_events(max_events)

        if include_events:
            event_list = include_events.split(",")
            config_builder = config_builder.include_events(event_list)

        if max_repos:
            config_builder = config_builder.max_repos(max_repos)

        # Apply strategy
        if strategy_override == "intelligence_guided":
            config_builder = config_builder.intelligence_guided()
        elif strategy_override == "multi_source":
            config_builder = config_builder.comprehensive()

        analysis_config = config_builder.build()

        # Create analyzer and run analysis
        analyzer = GitHubAnalyzer(token)

        # Progress callback for Rich UI
        def progress_callback(_current: int, _total: int, message: str) -> None:
            print(f"[dim]{message}[/dim]")

        # Perform analysis using library API
        result = analyzer.analyze_user(user, analysis_config, progress_callback)

        # Handle output
        if output:
            output_path = Path(output)
            with output_path.open("w") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"[green]✓ Results saved to {output}[/green]")
        else:
            # Print summary to console
            _print_analysis_summary(result)

    except InvalidTokenError as e:
        print(f"[red]Invalid GitHub token: {e}[/red]")
        print(
            "[yellow]Please check your token and try again, or run 'git-summary auth' to set up a new token[/yellow]"
        )
        raise typer.Exit(1)
    except UserNotFoundError as e:
        print(f"[red]User not found: {e}[/red]")
        raise typer.Exit(1)
    except ConfigurationError as e:
        print(f"[red]Configuration error: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        print("\n[yellow]Analysis cancelled by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("ai-summary")
def ai_summary(
    user: str = typer.Argument(None, help="GitHub username to analyze"),
    token: str | None = typer.Option(
        None,
        "--token",
        "-t",
        help="GitHub Personal Access Token (or set GITHUB_TOKEN env var)",
        envvar="GITHUB_TOKEN",
    ),
    model: str = typer.Option(
        "anthropic/claude-3-7-sonnet-latest",
        "--model",
        "-m",
        help="AI model to use (e.g., claude-3-7-sonnet-latest, groq/llama-3.1-70b-versatile, gpt-4o-mini)",
    ),
    persona: str = typer.Option(
        "tech analyst",
        "--persona",
        "-p",
        help="Analysis persona to use (tech_analyst, etc.)",
    ),
    days: int = typer.Option(7, "--days", "-d", help="Number of days to analyze"),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file (.md or .json format). If not specified, prints to stdout",
    ),
    max_events: int | None = typer.Option(
        500,
        "--max-events",
        help="Maximum number of events to process (default: 500 for better date range coverage)",
    ),
    token_budget: int = typer.Option(
        200,
        "--token-budget",
        help="Token budget for AI context gathering (200-1000 range)",
    ),
    estimate_cost: bool = typer.Option(
        False,
        "--estimate-cost",
        help="Show cost estimate without generating summary",
    ),
    repo: list[str] | None = typer.Option(
        None,
        "--repo",
        "-r",
        help="Filter events to specific repositories (e.g., --repo owner/repo-name). Can be used multiple times.",
    ),
) -> None:
    """Generate an AI-powered summary of GitHub activity.

    Examples:
        git-summary ai-summary username
        git-summary ai-summary username --repo owner/repo-name
        git-summary ai-summary username --repo repo1 --repo owner/repo2
        git-summary ai-summary username --days 30 --repo myorg/project
    """
    config = Config()

    # Validate CLI parameters
    _validate_cli_params(days, None, None, max_events)

    # Additional AI-specific validation
    if token_budget < 50 or token_budget > 1000:
        raise typer.BadParameter("--token-budget must be between 50 and 1000")

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

    print(f"[green]Generating AI-powered summary for user: {user}[/green]")
    print(f"[blue]Time range: Last {days} days[/blue]")
    print(f"[magenta]AI Model: {model}[/magenta]")
    print(f"[cyan]Persona: {persona}[/cyan]")
    if max_events:
        print(f"[yellow]Event limit: {max_events} events maximum[/yellow]")
    if output:
        print(f"[green]Output will be saved to: {output}[/green]")

    # Run the async AI analysis
    try:
        asyncio.run(
            _generate_ai_summary(
                user,
                token,
                model,
                persona,
                days,
                output,
                max_events,
                token_budget,
                estimate_cost,
                repo,
            )
        )
    except KeyboardInterrupt:
        print("\n[red]Operation cancelled by user[/red]")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


async def _generate_ai_summary(
    user: str,
    token: str,
    model: str,
    persona: str,
    days: int,
    output: str | None,
    max_events: int | None,
    token_budget: int,
    estimate_cost: bool,
    repo: list[str] | None,
) -> None:
    """Generate AI-powered summary of GitHub activity."""
    # Calculate date range
    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=days)

    console.print(
        Panel.fit(
            f"[bold magenta]AI-Powered GitHub Activity Summary[/bold magenta]\n\n"
            f"[white]User:[/white] [green]{user}[/green]\n"
            f"[white]Period:[/white] [blue]{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}[/blue]\n"
            f"[white]Duration:[/white] [yellow]{days} days[/yellow]\n"
            f"[white]AI Model:[/white] [magenta]{model}[/magenta]\n"
            f"[white]Persona:[/white] [cyan]{persona}[/cyan]",
            title="🤖 AI Analysis Configuration",
        )
    )

    # Initialize clients using unified architecture
    async with GitHubClient(token=token) as github_client:
        adaptive_discovery = AdaptiveRepositoryDiscovery(github_client)

        try:
            # Import here to avoid issues if AI dependencies aren't available
            from git_summary.ai.client import LLMClient

            # Initialize AI components
            llm_client = LLMClient(model=model)
            summarizer = ActivitySummarizer(
                github_client=github_client,
                llm_client=llm_client,
                default_token_budget=token_budget,
            )

            # Fetch events using adaptive system with progress indication
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

                # Use adaptive discovery for intelligent event fetching
                user_activity = await adaptive_discovery.analyze_user(
                    username=user,
                    days=days,
                    force_strategy=None,  # Let it auto-select optimal strategy
                )

                events = user_activity.events

                # Apply repository filtering if specified (client-side filtering)
                if repo:
                    original_count = len(events)
                    repo_names = [r.lower() for r in repo]
                    events = [
                        event
                        for event in events
                        if event.repo
                        and (
                            event.repo.name.lower() in repo_names
                            or (
                                event.repo.full_name
                                and event.repo.full_name.lower() in repo_names
                            )
                        )
                    ]
                    console.print(
                        f"[blue]→[/blue] Repository filter applied: {original_count} → {len(events)} events"
                    )

                # Apply max_events limit if specified
                if max_events and len(events) > max_events:
                    original_count = len(events)
                    events = events[:max_events]
                    console.print(
                        f"[blue]→[/blue] Limited to {max_events} events (from {original_count} total)"
                    )

                progress.update(
                    fetch_task,
                    description=f"✓ Fetched events for [green]{user}[/green]",
                )

            console.print(
                f"\n[green]✓[/green] Successfully fetched [cyan]{len(events)}[/cyan] events"
            )

            # Filter events to only include relevant types for AI analysis
            relevant_event_types = {
                "PushEvent",  # Commits
                "PullRequestEvent",  # Pull requests
                "PullRequestReviewEvent",  # PR reviews/comments
                "ReleaseEvent",  # Releases
                "IssueCommentEvent",  # Issue comments
                "PullRequestReviewCommentEvent",  # PR review comments
                "IssuesEvent",  # Issue creation, assignment, closing, labeling
                "CreateEvent",  # Branch/tag creation
                "DeleteEvent",  # Branch/tag deletion
                "GollumEvent",  # Wiki edits
            }

            original_count = len(events)
            events = [event for event in events if event.type in relevant_event_types]
            filtered_count = len(events)

            console.print(
                f"[blue]→[/blue] Filtered to [cyan]{filtered_count}[/cyan] relevant events "
                f"(commits, PRs, reviews, releases, comments, issues, branches)"
            )

            if original_count > filtered_count:
                console.print(
                    f"[dim]  Excluded {original_count - filtered_count} events "
                    f"(stars, forks, watches, etc.)[/dim]"
                )

            if not events:
                console.print(
                    f"[yellow]No relevant development events found for {user} in the last {days} days[/yellow]"
                )
                console.print(
                    "[dim]Try increasing the time range with --days or check if the user has recent development activity[/dim]"
                )
                return

            # Show cost estimate if requested
            if estimate_cost:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=False,
                ) as progress:
                    estimate_task = progress.add_task(
                        "Estimating AI processing cost..."
                    )

                    cost_info = await summarizer.estimate_cost(
                        cast("list[GitHubEventLike]", events),
                        persona_name=persona,
                        token_budget=token_budget,
                    )

                    progress.update(
                        estimate_task, description="✓ Cost estimation complete"
                    )

                _display_cost_estimate(cost_info)
                return

            # Generate AI summary with progress indication
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=False,
            ) as progress:
                ai_task = progress.add_task("Generating AI-powered summary...")

                # Generate the summary
                summary_result = await summarizer.generate_summary(
                    cast("list[GitHubEventLike]", events),
                    persona_name=persona,
                    token_budget=token_budget,
                    include_context_details=False,  # Streamlined output for CLI
                )

                # Add user info to the result for proper markdown generation
                summary_result["user"] = user

                progress.update(ai_task, description="✓ AI summary generation complete")

            console.print("[green]✓[/green] AI analysis complete!\n")

            # Display the AI summary
            _display_ai_summary(summary_result)

            # Save to file if requested
            if output:
                _save_ai_summary_to_file(summary_result, output)
                console.print(
                    f"\n[green]✓[/green] Summary saved to [cyan]{output}[/cyan]"
                )

        except ImportError:
            console.print(
                "[red]✗[/red] AI dependencies not available. Please install with AI support."
            )
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"\n[red]✗[/red] AI analysis failed: {e}")
            raise


def _display_cost_estimate(cost_info: dict[str, Any]) -> None:
    """Display cost estimation information."""
    estimate = cost_info["cost_estimate"]

    cost_content = (
        f"[white]Estimated Cost:[/white] [green]${estimate.get('total_cost', 0):.4f}[/green]\n"
        f"[white]Input Tokens:[/white] [cyan]{estimate.get('input_tokens', 0):,}[/cyan]\n"
        f"[white]Output Tokens:[/white] [cyan]{estimate.get('output_tokens', 0):,}[/cyan]\n"
        f"[white]Context Tokens:[/white] [yellow]{cost_info['context_tokens']:,}[/yellow]\n"
        f"[white]Total Events:[/white] [blue]{cost_info['total_events']}[/blue]\n"
        f"[white]Model:[/white] [magenta]{cost_info['model']}[/magenta]"
    )

    console.print(Panel(cost_content, title="💰 Cost Estimate", border_style="yellow"))

    # Event breakdown
    breakdown = cost_info["event_breakdown"]
    if any(breakdown.values()):
        breakdown_table = Table(
            title="📊 Event Breakdown", show_header=True, header_style="bold blue"
        )
        breakdown_table.add_column("Event Type", style="cyan")
        breakdown_table.add_column("Count", justify="right", style="green")

        for event_type, count in breakdown.items():
            if count > 0:
                breakdown_table.add_row(
                    event_type.replace("_", " ").title(), str(count)
                )

        console.print(breakdown_table)


def _display_ai_summary(summary_result: dict[str, Any]) -> None:
    """Display the AI-generated summary."""
    # Main summary panel
    summary_content = summary_result["summary"]

    console.print(
        Panel(
            summary_content,
            title=f"🤖 AI Summary ({summary_result['persona_used']})",
            border_style="magenta",
            padding=(1, 2),
        )
    )

    # Metadata panel
    metadata = summary_result["metadata"]
    metadata_content = (
        f"[white]Events Processed:[/white] [cyan]{metadata['total_events']}[/cyan]\n"
        f"[white]Tokens Used:[/white] [yellow]{metadata['tokens_used']}[/yellow]\n"
        f"[white]Active Repositories:[/white] [green]{len(metadata['repositories'])}[/green]\n"
        f"[white]Model Used:[/white] [magenta]{summary_result['model_used']}[/magenta]"
    )

    # Add line changes metrics if available
    if "line_changes" in metadata:
        line_metrics = metadata["line_changes"]
        metadata_content += (
            f"\n[white]Lines Added:[/white] [green]+{line_metrics['lines_added']}[/green]\n"
            f"[white]Lines Deleted:[/white] [red]-{line_metrics['lines_deleted']}[/red]\n"
            f"[white]Net Lines:[/white] [cyan]{line_metrics['net_lines']:+d}[/cyan]\n"
            f"[white]Files Changed:[/white] [blue]{line_metrics['files_changed']}[/blue]"
        )

    if metadata["repositories"]:
        repos_text = ", ".join(metadata["repositories"][:3])
        if len(metadata["repositories"]) > 3:
            repos_text += f" and {len(metadata['repositories']) - 3} more"
        metadata_content += f"\n[white]Repositories:[/white] [blue]{repos_text}[/blue]"

    console.print(
        Panel(metadata_content, title="📊 Analysis Metadata", border_style="blue")
    )


def _save_ai_summary_to_file(summary_result: dict[str, Any], output_path: str) -> None:
    """Save AI summary to file in appropriate format."""
    output_file = Path(output_path)

    if output_file.suffix.lower() == ".json":
        # Save as JSON
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary_result, f, indent=2, default=str)
    else:
        # Save as Markdown (default)
        if not output_file.suffix:
            output_file = output_file.with_suffix(".md")

        metadata = summary_result["metadata"]

        # Format date range properly
        date_range = metadata.get("date_range")
        if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            if hasattr(start_date, "strftime") and hasattr(end_date, "strftime"):
                formatted_period = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            else:
                formatted_period = str(date_range)
        else:
            formatted_period = metadata.get("date_range", "Unknown")

        # Build metadata section
        metadata_lines = [
            f"- **Events Processed:** {metadata['total_events']}",
            f"- **Tokens Used:** {metadata['tokens_used']}",
            f"- **Active Repositories:** {len(metadata['repositories'])}",
            f"- **Repositories:** {', '.join(metadata['repositories'])}",
        ]

        # Add line changes if available
        if "line_changes" in metadata:
            line_metrics = metadata["line_changes"]
            metadata_lines.extend(
                [
                    f"- **Lines Added:** +{line_metrics['lines_added']}",
                    f"- **Lines Deleted:** -{line_metrics['lines_deleted']}",
                    f"- **Net Lines:** {line_metrics['net_lines']:+d}",
                    f"- **Files Changed:** {line_metrics['files_changed']}",
                ]
            )

        metadata_section = "\n".join(metadata_lines)

        markdown_content = f"""# GitHub Activity Summary

**User:** {summary_result.get("user", "Unknown")}
**Period:** {formatted_period}
**AI Model:** {summary_result["model_used"]}
**Persona:** {summary_result["persona_used"]}

## AI Analysis

{summary_result["summary"]}

## Analysis Metadata

{metadata_section}

---
*Generated by git-summary AI-powered analysis*
"""

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)


async def _analyze_user_activity(
    user: str,
    token: str,
    days: int,
    output: str | None,
    max_events: int | None,
    include_events: str | None,
    strategy_override: str | None = None,
    max_repos: int | None = None,
) -> None:
    """Perform the actual GitHub activity analysis with robust error handling."""
    # Validate input parameters
    if days > 365:
        console.print("[red]Error: Analysis period cannot exceed 365 days[/red]")
        raise typer.Exit(1)

    if max_events and max_events < 1:
        console.print("[red]Error: max-events must be greater than 0[/red]")
        raise typer.Exit(1)

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

    # Validate GitHub token first
    console.print("[dim]Validating GitHub token...[/dim]")
    try:
        if not await validate_github_token(token):
            console.print("[red]Error: Invalid or expired GitHub token[/red]")
            console.print(
                "[yellow]Please check your token and try again, or run 'git-summary auth' to set up a new token[/yellow]"
            )
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error validating token: {e}[/red]")
        raise typer.Exit(1)

    # Set up signal handler for graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(_signum: int, _frame: Any) -> None:
        console.print(
            "\n[yellow]Cancelling analysis... Please wait for cleanup.[/yellow]"
        )
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    # Initialize resources with proper async context management
    try:
        async with asyncio.timeout(300):  # 5 minute timeout for entire analysis
            async with GitHubClient(token=token) as client:
                adaptive_discovery = AdaptiveRepositoryDiscovery(client)
                processor = EventProcessor()

                # Check for shutdown signal
                if shutdown_event.is_set():
                    console.print("[yellow]Analysis cancelled by user[/yellow]")
                    return

                await _perform_analysis(
                    adaptive_discovery,
                    processor,
                    user,
                    days,
                    output,
                    max_events,
                    include_events,
                    strategy_override,
                    max_repos,
                    shutdown_event,
                    start_date,
                    end_date,
                )

    except TimeoutError:
        console.print("[red]Analysis timed out after 5 minutes[/red]")
        console.print(
            "[yellow]Try reducing the analysis period with --days or limiting repositories with --max-repos[/yellow]"
        )
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis cancelled by user[/yellow]")
        raise typer.Exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        console.print(f"[red]Analysis failed: {e}[/red]")
        raise typer.Exit(1)


async def _perform_analysis(
    adaptive_discovery: AdaptiveRepositoryDiscovery,
    processor: EventProcessor,
    user: str,
    days: int,
    output: str | None,
    max_events: int | None,
    include_events: str | None,
    strategy_override: str | None,
    max_repos: int | None,
    shutdown_event: asyncio.Event,
    start_date: datetime,
    end_date: datetime,
) -> None:
    """Perform the core analysis logic with cancellation support."""
    try:
        # Check for shutdown signal before starting
        if shutdown_event.is_set():
            console.print("[yellow]Analysis cancelled by user[/yellow]")
            return

        # Perform adaptive analysis with progress indication
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("•"),
            TextColumn("[cyan]{task.fields[info]}[/cyan]"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            # Phase 1: User profiling
            profile_task = progress.add_task(
                f"Analyzing user profile for [green]{user}[/green]...", info="profiling"
            )

            # Phase 2: Adaptive analysis
            analysis_task = progress.add_task(
                "Running adaptive analysis...", info="analyzing"
            )

            # Check for cancellation before analysis
            if shutdown_event.is_set():
                console.print("[yellow]Analysis cancelled by user[/yellow]")
                return

            # Create progress callback to notify user of strategy decisions
            def strategy_callback(message: str) -> None:
                console.print(f"[cyan]→[/cyan] {message}")

            # Perform the adaptive analysis
            user_activity = await adaptive_discovery.analyze_user(
                username=user,
                days=days,
                force_strategy=strategy_override,
                progress_callback=strategy_callback,
            )

            # Validate analysis result
            if not user_activity:
                console.print("[red]Error: Analysis returned no results[/red]")
                raise typer.Exit(1)

            if not user_activity.events:
                console.print(
                    f"[yellow]No events found for user {user} in the last {days} days[/yellow]"
                )
                console.print(
                    "[dim]Try increasing the time range with --days or check if the user has recent activity[/dim]"
                )
                return

            # Note: max_events and max_repos would be handled by the adaptive discovery
            # system internally. For now, we apply max_events as a post-filter if specified.
            # max_repos is currently handled by the individual strategy implementations.
            if max_events and len(user_activity.events) > max_events:
                original_count = len(user_activity.events)
                user_activity.events = user_activity.events[:max_events]
                console.print(
                    f"[yellow]→[/yellow] Limited to {max_events} events (from {original_count} total)"
                )

            # TODO: Future enhancement - pass max_repos to adaptive discovery system
            _ = max_repos  # Suppress unused argument warning

            progress.update(
                profile_task,
                description=f"✓ Profiled [green]{user}[/green]",
                completed=True,
            )

            # Display strategy information with better error handling
            strategy_used = getattr(user_activity, "analysis_strategy", "unknown")
            if strategy_used == "unknown":
                console.print(
                    "[yellow]Warning: Analysis strategy could not be determined[/yellow]"
                )
            elif strategy_used == "fallback":
                console.print(
                    "[yellow]Warning: Using basic fallback strategy due to API limitations[/yellow]"
                )
            else:
                strategy_display = {
                    "intelligence_guided": "[blue]Intelligence-Guided Analysis[/blue] (optimized path)",
                    "multi_source": "[magenta]Comprehensive Multi-Source Discovery[/magenta] (complete coverage)",
                }.get(strategy_used, f"[yellow]{strategy_used}[/yellow]")
                console.print(f"\n[green]→[/green] Strategy: {strategy_display}")

            # Show automation classification if available
            if hasattr(user_activity, "profile") and user_activity.profile:
                classification = user_activity.profile.classification
                confidence = user_activity.profile.confidence_score
                if classification == "heavy-automation":
                    console.print(
                        f"[yellow]→[/yellow] User classified as automation (confidence: {confidence:.1%})"
                    )
                else:
                    console.print(
                        f"[cyan]→[/cyan] User classified as {classification} developer (confidence: {confidence:.1%})"
                    )

            events = user_activity.events

            progress.update(
                analysis_task,
                description=f"✓ Analyzed {len(user_activity.repositories)} repositories",
                info=f"{len(events)} events",
                completed=True,
            )

        # Display comprehensive analysis results
        console.print(
            f"\n[green]✓[/green] Successfully analyzed [cyan]{len(events)}[/cyan] events from [blue]{len(user_activity.repositories)}[/blue] repositories"
        )

        # Show repository coverage details
        if user_activity.repositories:
            console.print(
                f"[dim]  Repositories: {', '.join(user_activity.repositories[:5])}"
            )
            if len(user_activity.repositories) > 5:
                console.print(
                    f"[dim]  ... and {len(user_activity.repositories) - 5} more repositories"
                )

        # Show performance metrics if available
        if (
            hasattr(user_activity, "execution_time_ms")
            and user_activity.execution_time_ms
        ):
            execution_time = user_activity.execution_time_ms / 1000
            console.print(f"[dim]  Analysis completed in {execution_time:.2f} seconds")

        # Show strategy-specific stats if available
        if hasattr(user_activity, "analysis_stats") and user_activity.analysis_stats:
            stats = user_activity.analysis_stats
            if isinstance(stats, dict) and "api_calls_made" in stats:
                console.print(f"[dim]  API calls: {stats['api_calls_made']}")
            elif hasattr(stats, "api_calls_made"):
                console.print(f"[dim]  API calls: {stats.api_calls_made}")

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

    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis cancelled by user[/yellow]")
        raise
    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            console.print(f"[red]Rate limit exceeded: {error_msg}[/red]")
            console.print(
                "[yellow]Try again later or reduce the analysis scope with --days or --max-repos[/yellow]"
            )
        elif "not found" in error_msg.lower() or "404" in error_msg:
            console.print(f"[red]User or repository not found: {error_msg}[/red]")
            console.print(
                "[yellow]Please check the username and ensure it exists on GitHub[/yellow]"
            )
        elif "permission" in error_msg.lower() or "403" in error_msg:
            console.print(f"[red]Permission denied: {error_msg}[/red]")
            console.print(
                "[yellow]Please check your token permissions or try with a different token[/yellow]"
            )
        else:
            console.print(f"[red]Analysis failed: {error_msg}[/red]")
        raise


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
    table.add_row("GitHub Token", "✓ Yes" if info["has_token"] else "✗ No")

    if info["config_exists"]:
        table.add_row("File Permissions", info["config_file_permissions"] or "unknown")

    # Add AI API key status
    ai_keys = info["ai_api_keys"]
    for provider, has_key in ai_keys.items():
        status = "✓ Yes" if has_key else "✗ No"
        table.add_row(f"{provider.title()} API Key", status)

    print(table)

    if not info["has_token"]:
        print()
        print(
            "[yellow]No GitHub token found. Run [bold]git-summary auth[/bold] to set up authentication.[/yellow]"
        )

    if not any(ai_keys.values()):
        print()
        print(
            "[yellow]No AI API keys found. Run [bold]git-summary ai-auth <provider>[/bold] to set up AI API keys.[/yellow]"
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


@app.command("ai-auth")
def ai_auth(
    provider: str = typer.Argument(
        ..., help="AI provider (openai, anthropic, google, groq)"
    ),
) -> None:
    """Manage AI API keys for different providers."""
    config = Config()

    if provider.lower() not in ["openai", "anthropic", "google", "groq"]:
        print(
            f"[red]Error: Unknown provider '{provider}'. Use: openai, anthropic, google, or groq[/red]"
        )
        raise typer.Exit(1)

    provider = provider.lower()

    print(f"[bold cyan]AI API Key Setup - {provider.title()}[/bold cyan]")
    print()

    current_key = config.get_ai_api_key(provider)
    if current_key:
        print(f"[green]✓[/green] You already have a {provider} API key stored")

        if not Confirm.ask("Would you like to replace it with a new key?"):
            return

    # Provider-specific instructions
    instructions = {
        "openai": "You can get an API key from: [link]https://platform.openai.com/api-keys[/link]",
        "anthropic": "You can get an API key from: [link]https://console.anthropic.com/[/link]",
        "google": "You can get an API key from: [link]https://console.cloud.google.com/[/link]",
        "groq": "You can get an API key from: [link]https://console.groq.com/keys[/link]",
    }

    print(instructions[provider])
    print()

    api_key = Prompt.ask(f"[cyan]Enter your {provider.title()} API key", password=True)

    if not api_key:
        print("[red]No API key provided[/red]")
        return

    config.set_ai_api_key(provider, api_key)
    print(f"[green]✓[/green] {provider.title()} API key setup complete!")


@app.command("ai-auth-status")
def ai_auth_status() -> None:
    """Show current AI API key status."""
    config = Config()
    ai_keys = config.list_ai_api_keys()

    table = Table(title="AI API Key Status")
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="green")

    for provider, has_key in ai_keys.items():
        status = "✓ Configured" if has_key else "✗ Not configured"
        table.add_row(provider.title(), status)

    print(table)

    if not any(ai_keys.values()):
        print()
        print(
            "[yellow]No AI API keys configured. Run [bold]git-summary ai-auth <provider>[/bold] to set up API keys.[/yellow]"
        )


@app.command("ai-auth-remove")
def ai_auth_remove(
    provider: str = typer.Argument(
        ..., help="AI provider (openai, anthropic, google, groq)"
    ),
) -> None:
    """Remove stored AI API key for a provider."""
    config = Config()

    if provider.lower() not in ["openai", "anthropic", "google", "groq"]:
        print(
            f"[red]Error: Unknown provider '{provider}'. Use: openai, anthropic, google, or groq[/red]"
        )
        raise typer.Exit(1)

    provider = provider.lower()

    if not config.get_ai_api_key(provider):
        print(f"[yellow]No {provider} API key is currently stored[/yellow]")
        return

    if Confirm.ask(
        f"[red]Are you sure you want to remove the {provider} API key?[/red]"
    ):
        config.remove_ai_api_key(provider)
    else:
        print("API key removal cancelled")


@app.command("personas")
def list_personas(
    show_sources: bool = typer.Option(
        False, "--sources", "-s", help="Show persona sources (built-in, package, user)"
    ),
) -> None:
    """List all available AI personas."""
    try:
        manager = PersonaManager()
        personas_by_type = manager.list_personas_by_type()

        console.print("\n[bold]Available AI Personas:[/bold]\n")

        if show_sources:
            # Show detailed breakdown by source
            if personas_by_type["built_in"]:
                console.print("[bold cyan]Built-in Personas:[/bold cyan]")
                for persona in personas_by_type["built_in"]:
                    console.print(f"  • {persona.name}: {persona.description}")
                console.print()

            if personas_by_type["package_yaml"]:
                console.print("[bold green]Package YAML Personas:[/bold green]")
                for persona in personas_by_type["package_yaml"]:
                    console.print(f"  • {persona.name}: {persona.description}")
                console.print()

            if personas_by_type["user_yaml"]:
                console.print("[bold magenta]Your Custom Personas:[/bold magenta]")
                for persona in personas_by_type["user_yaml"]:
                    console.print(f"  • {persona.name}: {persona.description}")
                    console.print(f"    [dim]📁 {persona.yaml_path}[/dim]")
                console.print()
        else:
            # Standard table view
            table = Table(show_header=True, header_style="bold blue")
            table.add_column("Name", style="cyan")
            table.add_column("Source", style="yellow")
            table.add_column("Description", style="white")

            # Add built-in personas
            for persona in personas_by_type["built_in"]:
                table.add_row(persona.name, "Built-in", persona.description)

            # Add package YAML personas
            for persona in personas_by_type["package_yaml"]:
                table.add_row(persona.name, "Package", persona.description)

            # Add user YAML personas
            for persona in personas_by_type["user_yaml"]:
                table.add_row(persona.name, "Custom", persona.description)

            console.print(table)

        # Show usage hint
        console.print(
            "[dim]💡 Use personas with: [yellow]git-summary ai-summary --persona <name>[/yellow][/dim]"
        )

        # Show total count and breakdown
        total_personas = len(personas_by_type["built_in"]) + len(
            personas_by_type["yaml"]
        )
        user_count = len(personas_by_type["user_yaml"])

        console.print(f"[dim]📊 Total: {total_personas} personas available[/dim]")
        if user_count > 0:
            console.print(
                f"[dim]👤 Custom personas: {user_count} in ~/.git-summary/personas/[/dim]"
            )

    except Exception as e:
        console.print(f"[red]❌ Error listing personas: {e}[/red]")
        raise typer.Exit(1)


@app.command("create-persona")
def create_persona_template(
    name: str = typer.Argument(..., help="Name for the new persona"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output file path"),
    template_type: str = typer.Option(
        "basic", "--type", "-t", help="Template type: basic, technical"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing file"),
) -> None:
    """Create a new YAML persona template."""
    try:
        manager = PersonaManager()

        # Handle output path
        output_path = Path(output) if output else None

        # Check if file exists and handle force flag
        if output_path and output_path.exists() and not force:
            console.print(f"[red]❌ File already exists: {output_path}[/red]")
            console.print("[yellow]💡 Use --force to overwrite existing files[/yellow]")
            raise typer.Exit(1)

        # Remove existing file if force is used
        if force and output_path and output_path.exists():
            output_path.unlink()
            console.print(f"[yellow]🗑️  Removed existing file: {output_path}[/yellow]")

        # Validate template type
        if template_type not in ["basic", "technical"]:
            console.print(f"[red]❌ Invalid template type: {template_type}[/red]")
            console.print("[yellow]💡 Available types: basic, technical[/yellow]")
            raise typer.Exit(1)

        created_path = manager.create_persona_template(name, output_path, template_type)

        console.print(
            f"[green]✅ Created persona template: [cyan]{created_path}[/cyan][/green]"
        )
        console.print("[yellow]📝 Edit the file to customize the persona[/yellow]")
        console.print(
            f"[dim]💡 Then use it with: [yellow]git-summary ai-summary --persona '{name.lower()}'[/yellow][/dim]"
        )

        # Show template info
        template_info = {
            "basic": "General-purpose persona with flexible analysis sections",
            "technical": "Engineering-focused persona with technical depth",
        }
        console.print(
            f"[dim]📋 Template type: {template_type} - {template_info[template_type]}[/dim]"
        )

    except ValueError as e:
        console.print(f"[red]❌ {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Failed to create persona template: {e}[/red]")
        raise typer.Exit(1)


@app.command("persona-info")
def persona_info(
    name: str = typer.Argument(..., help="Name of the persona to inspect"),
) -> None:
    """Show detailed information about a persona."""
    try:
        manager = PersonaManager()
        info = manager.get_persona_info(name)

        console.print(
            f"\n[bold cyan]📋 Persona Information: {info['name']}[/bold cyan]\n"
        )

        # Basic info table
        basic_table = Table(show_header=False, box=None)
        basic_table.add_column("Property", style="yellow")
        basic_table.add_column("Value", style="white")

        basic_table.add_row("Name", info["name"])
        basic_table.add_row("Type", info["type"].title())
        basic_table.add_row("Description", info["description"])

        # Add YAML-specific information
        if info["type"] == "yaml":
            basic_table.add_row("File Path", info["yaml_path"])
            basic_table.add_row("Version", info["version"])
            basic_table.add_row("Author", info["author"])
            basic_table.add_row("Sections", str(info["sections"]))
            basic_table.add_row("Max Words", str(info["max_words"]))
            basic_table.add_row("Tone", info["tone"].title())
            basic_table.add_row("Audience", info["audience"].title())

        console.print(basic_table)

        # Usage example
        persona_name_for_cli = info["name"].lower().replace(" ", "_")
        console.print(
            f"\n[dim]💡 Usage: [yellow]git-summary ai-summary --persona '{persona_name_for_cli}' username[/yellow][/dim]"
        )

    except ValueError as e:
        console.print(f"[red]❌ Persona not found: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error getting persona info: {e}[/red]")
        raise typer.Exit(1)


@app.command("reload-personas")
def reload_personas() -> None:
    """Reload YAML personas from disk (useful for development)."""
    try:
        manager = PersonaManager()

        console.print("[yellow]🔄 Reloading YAML personas...[/yellow]")

        # Count before reload
        before_count = len(manager.list_personas_by_type()["yaml"])

        manager.reload_yaml_personas()

        # Count after reload
        after_count = len(manager.list_personas_by_type()["yaml"])

        console.print(f"[green]✅ Reloaded {after_count} YAML personas[/green]")

        if after_count != before_count:
            if after_count > before_count:
                console.print(
                    f"[cyan]📈 Found {after_count - before_count} new personas[/cyan]"
                )
            else:
                console.print(
                    f"[yellow]📉 {before_count - after_count} personas were removed or failed to load[/yellow]"
                )

        console.print(
            "[dim]💡 Run [yellow]git-summary personas[/yellow] to see all available personas[/dim]"
        )

    except Exception as e:
        console.print(f"[red]❌ Error reloading personas: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
