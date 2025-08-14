# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GitHub Activity Tracker - a comprehensive tool for querying the GitHub API to extract, analyze, and summarize user activity across repositories and projects. The system creates automated summaries of GitHub user activity including commits, issues, pull requests, and other events.

## Current State

This is a modern Python 3.12+ project using UV package manager, following best practices for Python development.

**✅ MILESTONE 3 COMPLETED**: Comprehensive Adaptive GitHub Analysis System with intelligent strategy selection, automation detection, and multi-source discovery. Overcomes GitHub's 300-event limitation through advanced analysis algorithms.

## Core Architecture (Implemented)

### Adaptive Analysis System (NEW)
- **AutomationDetector**: User pattern classification with 4-factor analysis
- **IntelligenceGuidedAnalyzer**: Fast analysis for 98% of users (15-30 API calls)
- **MultiSourceDiscovery**: Comprehensive analysis for power users (60-120 API calls)
- **AdaptiveRepositoryDiscovery**: Main coordinator with circuit breakers and fallbacks

### Data Collection Layer
- GitHub API integration using OAuth2 or Personal Access Tokens
- Multi-endpoint querying (`/user/events`, `/repos/{owner}/{repo}/events`, `/search/commits`, etc.)
- Rate limiting compliance with intelligent REST vs Search API balancing
- **NEW**: Multi-source data correlation across owned repos, user events, and commit search

### Data Processing Layer
- Time-based activity summarization (daily, weekly, custom ranges)
- Event categorization and filtering by type
- Repository-based activity grouping and filtering
- **NEW**: Repository scoring algorithm with weighted event types
- **NEW**: Composite key event deduplication across data sources
- **NEW**: Priority-based repository ranking with multi-source scoring

### Output Generation
- Structured JSON export with standardized schema
- Daily rollups and time-based summaries
- Flexible filtering capabilities (date ranges, repositories, event types)
- **NEW**: Analysis strategy metadata and performance statistics

## Key Technical Requirements

### API Integration
- Support multiple GitHub API endpoints for comprehensive data collection
- Handle authentication with Personal Access Tokens and OAuth2
- Implement proper error handling for token validation, rate limits, and network errors

### Data Structure
The planned output follows this JSON schema:
```json
{
  "user": "username",
  "period": {"start": "ISO8601", "end": "ISO8601"},
  "summary": {"total_events": 0, "repositories_active": 0, "event_breakdown": {}},
  "daily_rollups": [],
  "repository_breakdown": {},
  "detailed_events": []
}
```

### Performance Considerations
- Rate limiting compliance with GitHub API
- Efficient pagination for large datasets
- Caching for frequently accessed data
- Batch processing capabilities

## Development Setup

### Package Management
- Uses **UV** as the package manager for fast dependency resolution and virtual environment management
- Project configuration in `pyproject.toml` following modern Python standards
- Uses **hatchling** as the build backend for PyPI releases

### Code Quality Tools
- **Ruff** for both linting and formatting (replaces black, flake8, isort, etc.)
- Pre-commit hooks for automated code quality checks
- GitHub Actions for CI/CD pipeline

### Common Commands
```bash
# Install dependencies and create virtual environment
uv sync

# Setup authentication (interactive)
uv run git-summary auth

# Check authentication status
uv run git-summary auth-status

# Basic usage (interactive mode if no token stored)
uv run git-summary summary username

# Full command with options
uv run git-summary summary username --days 30 --output results.json

# Using environment variable for token
GITHUB_TOKEN=your_token uv run git-summary summary username

# NEW: Adaptive analysis with intelligent strategy selection
uv run git-summary summary username --days 30                    # Auto-detects optimal strategy
uv run git-summary summary username --comprehensive              # Force multi-source analysis
uv run git-summary summary username --max-repos 15              # Limit repositories for focus
uv run git-summary summary username --force-strategy multi_source  # Override for testing

# AI-powered analysis with repository filtering
uv run git-summary ai-summary username --repo owner/repo-name
uv run git-summary ai-summary username --repo myorg/frontend --repo myorg/backend
uv run git-summary ai-summary username --days 30 --repo myproject/api --persona "Tech Analyst"

# Development commands
uv run ruff format
uv run ruff check
uv run ruff check --fix
uv run pytest
uv build

# Code quality analysis
uv run radon cc src/                    # Cyclomatic complexity analysis
uv run radon mi src/                    # Maintainability index (A/B/C rankings)
uv run radon raw src/                   # Raw metrics (LOC, comments, etc.)
uv run vulture src/                     # Dead code detection
uv run vulture src/ --min-confidence 90  # High-confidence dead code only
```

### Development Guidelines

1. **Python Version**: Requires Python 3.12+

2. **Authentication**: Implement secure token storage and management from the start

3. **API Coverage**: Ensure comprehensive coverage of GitHub event types:
   - Commits and commit comments
   - Issues (created, commented, assigned, closed, labeled)
   - Pull requests (created, reviewed, merged, commented)
   - Branch management (creation, deletion)
   - Code reviews and PR feedback
   - Documentation (wiki edits)
   - Repository events (releases, significant activities only)

4. **Output Flexibility**: Design the system to support multiple output formats and verbosity levels

5. **Error Handling**: Implement robust error handling for API failures, network issues, and data validation

6. **Code Quality Standards**: Before finalizing any significant code changes, run quality analysis tools to maintain codebase health:
   - Use `radon cc` to identify functions with high cyclomatic complexity (aim for Rank A/B, avoid C+ complexity)
   - Check maintainability with `radon mi` (target A-rank maintainability index)
   - Clean up dead code identified by `vulture` (focus on high-confidence findings)
   - Track quality trends over time - complexity should not increase without justification

## CLI Features

### Interactive Authentication
- **Token Storage**: Securely stores GitHub Personal Access Tokens in `~/.git-summary/config.json` with 600 permissions
- **Interactive Setup**: `git-summary auth` provides guided token setup
- **Multiple Token Sources**: Supports CLI args, environment variables, or stored tokens
- **Token Management**: Commands for checking status and removing stored tokens

### Smart Interactive Mode
- **Username Prompts**: Asks for username if not provided as argument
- **Token Prompts**: Interactive token entry if none found in storage/environment
- **Save Preference**: Option to save tokens for future use during interactive flow

### MVP Filtering (Placeholders)
- `--include-events`: Placeholder for event type filtering (shows warning)
- `--exclude-repos`: Placeholder for repository exclusion (shows warning)
- These options are present but not implemented - they display MVP warnings

### Available Commands
- `git-summary summary [username]` - **NEW**: Adaptive analysis with intelligent strategy selection
  - `--comprehensive` - Force multi-source comprehensive analysis
  - `--max-repos INTEGER` - Limit repositories analyzed (adaptive default)
  - `--force-strategy STRATEGY` - Override strategy: intelligence_guided | multi_source
- `git-summary ai-summary [username]` - AI-powered analysis with repository filtering support
- `git-summary auth` - Interactive token setup and management
- `git-summary auth-status` - Show current authentication status with table
- `git-summary auth-remove` - Remove stored token with confirmation
- `git-summary version` - Show version information

## Security Considerations

- Never commit authentication tokens to the repository
- Implement minimal required scope permissions
- Include audit logging for API access
- Follow data privacy best practices for user activity data

## Git Workflow and Ticket Management

### CRITICAL: Always Commit Before Closing Tickets
- **NEVER close a GitHub ticket manually** unless there was a mistake or special circumstances
- **ALWAYS commit and push changes before closing tickets**
- Every ticket closure should reference a commit that implements the work
- This ensures traceability and proper version control of all changes

### Commit Guidelines
- All new files must be added to git (`git add`) before committing
- Commit messages should be descriptive and reference the ticket being completed
- Use the standard commit format with co-author attribution:
  ```
  Brief description of changes

  🤖 Generated with [Claude Code](https://claude.ai/code)

  Co-Authored-By: Claude <noreply@anthropic.com>
  ```
- Always push commits to the remote repository after committing

### Workflow
1. Complete implementation and testing
2. Add all new/modified files to git
3. Create commit with descriptive message
4. Push to remote repository
5. Close ticket with reference to the commit

## Git Worktree Development

For parallel development on multiple GitHub issues, use git worktrees to work on independent features simultaneously.

### Setup Worktrees
```bash
# Create worktrees directory (if not exists)
mkdir -p worktrees

# Create worktrees for parallel development (in local worktrees/ folder)
git worktree add worktrees/public-events-fetcher -b feature/public-events-fetcher
git worktree add worktrees/smart-coordinator -b feature/smart-coordinator
git worktree add worktrees/event-processing -b feature/event-processing

# List all worktrees
git worktree list
```

### Directory Structure
```
git-summary/                              # Main worktree (master branch)
├── worktrees/                            # Local worktrees folder (gitignored)
│   ├── public-events-fetcher/            # Feature worktree
│   │   ├── src/git_summary/
│   │   └── ...
│   ├── smart-coordinator/                # Feature worktree
│   └── event-processing/                 # Feature worktree
├── src/git_summary/
└── ...
```

### Parallel Development Workflow
1. **Main worktree**: Keep on `master` for integration work (Issue #14)
2. **Feature worktrees**: Work on independent issues:
   - `feature/public-events-fetcher` - Issue #11 (PublicEventsFetcher)
   - `feature/smart-coordinator` - Issue #12 (Smart Coordinator)
   - `feature/event-processing` - Issue #13 (Event Processing)

### Branch Naming Convention
- Use `feature/issue-name` pattern aligned with GitHub issue titles
- Keep branch names descriptive and consistent
- Example: `feature/public-events-fetcher` for Issue #11

### Dependency Management
```bash
# Share commits between worktrees when needed
cd worktrees/smart-coordinator
git cherry-pick feature/public-events-fetcher~1..feature/public-events-fetcher
```

### Integration Process
```bash
# Merge features back to main (from main worktree)
git merge feature/public-events-fetcher
git merge feature/smart-coordinator
git merge feature/event-processing

# Push all branches
git push origin master
git push origin feature/public-events-fetcher
git push origin feature/smart-coordinator
git push origin feature/event-processing
```

### Cleanup When Done
```bash
# Remove worktrees after feature completion
git worktree remove worktrees/public-events-fetcher
git worktree remove worktrees/smart-coordinator
git worktree remove worktrees/event-processing

# Delete feature branches if no longer needed
git branch -d feature/public-events-fetcher
git branch -d feature/smart-coordinator
git branch -d feature/event-processing
```

### Reference Documentation
- [Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)
- [Claude Code Common Workflows](https://docs.anthropic.com/en/docs/claude-code/common-workflows)

- consult with guilfoyle just after your code lands and iterate if needed. or pause and ask for input from a human
