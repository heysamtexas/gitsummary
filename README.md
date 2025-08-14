# git-summary

> A comprehensive CLI tool for analyzing and summarizing GitHub user activity

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![Built with UV](https://img.shields.io/badge/built%20with-uv-green)](https://docs.astral.sh/uv/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Extract, analyze, and summarize GitHub user activity across repositories and projects. Get comprehensive insights into commits, issues, pull requests, and other GitHub events with beautiful CLI output.

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- [UV package manager](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

#### Option 1: Using UV (Recommended)
```bash
# Clone the repository
git clone https://github.com/samtexas/git-summary.git
cd git-summary

# Install dependencies
uv sync

# Set up authentication
uv run git-summary auth
```

#### Option 2: Using pip
```bash
# Clone the repository
git clone https://github.com/samtexas/git-summary.git
cd git-summary

# Install in development mode
pip install -e .

# Set up authentication
git-summary auth
```

### First Run

1. **Set up GitHub authentication** (one time setup):
   ```bash
   uv run git-summary auth
   ```

2. **Analyze a GitHub user**:
   ```bash
   uv run git-summary summary octocat
   ```

## 📖 Usage

### Basic Commands

```bash
# Analyze a user's activity (last 7 days)
git-summary summary username

# Analyze with custom time range
git-summary summary username --days 30

# Save output to file
git-summary summary username --output results.json

# Interactive mode - prompts for username and token if needed
git-summary summary
```

### 🤖 AI-Powered Analysis

Generate intelligent, narrative summaries of GitHub activity with comprehensive engagement tracking. AI analysis includes commits, pull requests, issues, reviews, comments, branch management, and documentation contributions across all repositories or filtered to specific projects:

```bash
# Technical analysis for engineers
git-summary ai-summary username --persona "Tech Analyst"

# Business-focused insights for stakeholders
git-summary ai-summary username --persona "Product Manager"

# Engaging first-person narratives
git-summary ai-summary username --persona "Ghost Writer"

# Team collaboration insights
git-summary ai-summary username --persona "Team Lead"

# Data-driven metrics analysis
git-summary ai-summary username --persona "Data Analyst"

# Focus analysis on specific repositories
git-summary ai-summary username --repo owner/repo-name

# Analyze multiple repositories
git-summary ai-summary username --repo myorg/frontend --repo myorg/backend

# Combine with time range and persona
git-summary ai-summary username --days 30 --repo myproject/api --persona "Tech Analyst"
```

#### AI Command Options

```bash
git-summary ai-summary [USERNAME] [OPTIONS]

Options:
  --persona, -p TEXT       AI persona for analysis style (default: "Tech Analyst")
  --model, -m TEXT         AI model to use (default: "claude-3-7-sonnet-latest")
  --days, -d INTEGER       Number of days to analyze (default: 7)
  --output, -o TEXT        Output file (.md or .json format)
  --repo, -r TEXT          Filter events to specific repositories (can be used multiple times)
  --max-events INTEGER     Limit number of events processed
  --token-budget INTEGER   AI token budget for analysis (default: 8000)
  --estimate-cost          Show cost estimate before processing
```

#### Available Personas

| Persona | Focus | Best For |
|---------|-------|----------|
| **Tech Analyst** | Technical depth, code quality | Engineering teams, code reviews |
| **Product Manager** | Business impact, user value | Stakeholder updates, roadmaps |
| **Ghost Writer** | Personal narratives, storytelling | Portfolios, blog posts, retrospectives |
| **Team Lead** | Collaboration, team health | Team management, performance reviews |
| **Data Analyst** | Metrics, trends, quantitative insights | KPI tracking, trend analysis |

👉 **[Complete AI Personas Guide](docs/personas.md)** - Learn how to create custom personas

### Authentication Commands

```bash
# Set up GitHub Personal Access Token (interactive)
git-summary auth

# Check current authentication status
git-summary auth-status

# Remove stored token
git-summary auth-remove

# Set up AI API keys for analysis features
git-summary ai-auth anthropic    # For Claude models
git-summary ai-auth openai       # For GPT models
git-summary ai-auth groq         # For Groq models

# Check AI API key status
git-summary ai-auth-status

# Remove AI API key
git-summary ai-auth-remove anthropic

# Persona management
git-summary personas              # List available personas
git-summary personas --sources    # Show persona sources
git-summary create-persona "Custom Name" --type basic
git-summary reload-personas       # Reload custom personas
git-summary persona-info "Persona Name"
```

### All Available Options

```bash
git-summary summary [USERNAME] [OPTIONS]

Arguments:
  USERNAME    GitHub username to analyze (optional, will prompt if not provided)

Options:
  --token, -t TEXT         GitHub Personal Access Token
  --days, -d INTEGER       Number of days to analyze (default: 7)
  --output, -o TEXT        Output file (JSON format)
  --include-events TEXT    [Coming Soon] Event types to include
  --exclude-repos TEXT     [Coming Soon] Repositories to exclude
  --help                   Show help message
```

## 🔑 GitHub Authentication

### Creating a Personal Access Token

1. Go to [GitHub Settings > Developer settings > Personal access tokens](https://github.com/settings/tokens)
2. Click "Generate new token (classic)"
3. Select scopes:
   - `public_repo` (for public repositories)
   - `repo` (if you need access to private repositories)
4. Copy the generated token
5. Run `git-summary auth` and paste the token when prompted

### Token Storage

Tokens are securely stored in `~/.git-summary/config.json` with restricted file permissions (600). The file is only readable by your user account.

## 📊 Output Format

The tool outputs structured JSON data with the following schema:

```json
{
  "user": "username",
  "period": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-08T00:00:00Z"
  },
  "summary": {
    "total_events": 42,
    "repositories_active": 5,
    "event_breakdown": {
      "PushEvent": 15,
      "IssuesEvent": 8,
      "PullRequestEvent": 12
    }
  },
  "daily_rollups": [...],
  "repository_breakdown": {...},
  "detailed_events": [...]
}
```

## 🛠️ Development

### Setup Development Environment

```bash
# Clone and setup
git clone https://github.com/samtexas/git-summary.git
cd git-summary

# Install with development dependencies
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest

# Format code
uv run ruff format

# Lint code
uv run ruff check
```

### Project Structure

```
git-summary/
├── src/git_summary/          # Main package
│   ├── cli.py               # Command-line interface
│   ├── config.py            # Configuration and token management
│   └── github_client.py     # GitHub API client
├── tests/                   # Test suite
├── pyproject.toml          # Project configuration
└── README.md              # This file
```

### Code Quality

This project uses modern Python tooling:
- **UV**: Fast package management and virtual environments
- **Ruff**: Lightning-fast linting and formatting
- **MyPy**: Static type checking
- **Pytest**: Testing framework
- **Pre-commit**: Automated code quality checks

## 🔧 Configuration

### Environment Variables

```bash
# Set token via environment variable (alternative to stored config)
export GITHUB_TOKEN=your_token_here
git-summary summary username

# Or create a .env file
echo "GITHUB_TOKEN=your_token_here" > .env
```

### Configuration File Location

- **Config directory**: `~/.git-summary/`
- **Config file**: `~/.git-summary/config.json`
- **Custom personas**: `~/.git-summary/personas/*.yaml`
- **Permissions**: 700 (directory), 600 (files)

### Custom Personas

Create your own AI analysis styles by adding YAML files to `~/.git-summary/personas/`:

```bash
# Create a custom persona template
git-summary create-persona "Executive Briefer" --type basic

# Edit the YAML file to customize behavior
nano ~/.git-summary/personas/executive_briefer.yaml

# Reload to activate your changes
git-summary reload-personas

# Use your custom persona
git-summary ai-summary username --persona "Executive Briefer"
```

See the **[Complete Personas Guide](docs/personas.md)** for detailed configuration options.

## 🚨 Troubleshooting

### Common Issues

**"No GitHub token found"**
- Run `git-summary auth` to set up authentication
- Or set the `GITHUB_TOKEN` environment variable
- Or use `--token your_token` flag

**"API rate limit exceeded"**
- Authenticated requests have higher rate limits (5,000/hour vs 60/hour)
- Wait for the rate limit to reset
- Consider analyzing smaller time ranges

**"Permission denied" errors**
- Check that your token has the required scopes (`public_repo` or `repo`)
- Regenerate token if it's expired

**"No AI API key found" (for ai-summary command)**
- Run `git-summary ai-auth anthropic` to set up Claude API key
- Or `git-summary ai-auth openai` for OpenAI GPT models
- Or `git-summary ai-auth groq` for Groq models
- Check status with `git-summary ai-auth-status`

**"Persona not found" errors**
- Run `git-summary personas` to see available personas
- Check spelling and use exact persona names
- For custom personas, ensure YAML files are in `~/.git-summary/personas/`
- Run `git-summary reload-personas` after adding/editing custom personas

### Getting Help

```bash
# Show help for any command
git-summary --help
git-summary summary --help
git-summary auth --help

# Check authentication status
git-summary auth-status
```

## 📋 Supported GitHub Events

The tool analyzes all GitHub event types:

- **Code Events**: Pushes, commits, branches, tags
- **Issues**: Created, updated, closed, commented
- **Pull Requests**: Opened, closed, merged, reviewed
- **Repository Events**: Stars, forks, watches, releases
- **Collaboration**: Comments, reviews, mentions
- **Project Management**: Project boards, milestones

### AI Analysis Coverage

The AI-powered analysis (`ai-summary`) focuses on high-value development activities:

- **Code Development**: Commits, branches, releases
- **Issue Management**: Issue creation, assignment, closing, labeling
- **Pull Request Workflow**: PR creation, reviews, comments
- **Documentation**: Wiki edits and updates
- **Project Collaboration**: Issue discussions and PR feedback

*Note: Low-effort activities like starring and forking repositories are excluded from AI analysis to focus on meaningful contributions.*

## 🎯 Use Cases

### Personal Productivity
- Track your daily coding activity
- Generate reports for standups or reviews
- Monitor contribution patterns across projects
- Focus analysis on specific repositories for project-specific insights

### Team Management
- Analyze team member contributions
- Track project activity and engagement
- Generate activity reports for stakeholders
- Compare activity across different repositories or projects

### Project Analytics
- Monitor repository health and activity
- Track contributor engagement over time
- Analyze development velocity and patterns
- Get focused insights by filtering to specific repositories

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to our GitHub repository.

## 🔗 Links

- [GitHub Repository](https://github.com/samtexas/git-summary)
- [Issue Tracker](https://github.com/samtexas/git-summary/issues)
- [GitHub API Documentation](https://docs.github.com/en/rest)
