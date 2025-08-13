# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GitHub Activity Tracker - a comprehensive tool for querying the GitHub API to extract, analyze, and summarize user activity across repositories and projects. The system creates automated summaries of GitHub user activity including commits, issues, pull requests, and other events.

## Current State

This is a modern Python 3.12+ project using UV package manager, following best practices for Python development.

## Core Architecture (As Planned)

### Data Collection Layer
- GitHub API integration using OAuth2 or Personal Access Tokens
- Multi-endpoint querying (`/user/events`, `/repos/{owner}/{repo}/events`, etc.)
- Rate limiting compliance and pagination handling

### Data Processing Layer
- Time-based activity summarization (daily, weekly, custom ranges)
- Event categorization and filtering by type
- Repository-based activity grouping
- Comprehensive event extraction to ensure no user activity is missed

### Output Generation
- Structured JSON export with standardized schema
- Daily rollups and time-based summaries
- Flexible filtering capabilities (date ranges, repositories, event types)

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

# Development commands
uv run ruff format
uv run ruff check
uv run ruff check --fix
uv run pytest
uv build
```

### Development Guidelines

1. **Python Version**: Requires Python 3.12+

2. **Authentication**: Implement secure token storage and management from the start

3. **API Coverage**: Ensure comprehensive coverage of GitHub event types:
   - Commits and commit comments
   - Issues (created, commented, assigned, closed)
   - Pull requests (created, reviewed, merged, commented)
   - Repository events (stars, forks, watches)
   - Code reviews and project board activities

4. **Output Flexibility**: Design the system to support multiple output formats and verbosity levels

5. **Error Handling**: Implement robust error handling for API failures, network issues, and data validation

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
- `git-summary summary [username]` - Main analysis command with full interactivity
- `git-summary auth` - Interactive token setup and management
- `git-summary auth-status` - Show current authentication status with table
- `git-summary auth-remove` - Remove stored token with confirmation
- `git-summary version` - Show version information

## Security Considerations

- Never commit authentication tokens to the repository
- Implement minimal required scope permissions
- Include audit logging for API access
- Follow data privacy best practices for user activity data
