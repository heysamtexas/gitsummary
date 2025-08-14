# Git-Summary Documentation

Welcome to the git-summary documentation! This directory contains comprehensive guides for using and extending git-summary.

## 📚 Available Documentation

### 🤖 AI Features
- **[AI Personas Guide](personas.md)** - Complete guide to AI personas, creating custom personas, and YAML configuration

### 📋 Technical Documentation
- **[PRD: GitHub API Integration](PRD-GitHub-API-Integration.md)** - Product Requirements Document for GitHub API integration

## 🚀 Quick Links

### For Users
- **Getting Started**: See the main [README](../README.md) for installation and basic usage
- **AI Analysis**: Use `git-summary ai-summary username --persona "Ghost Writer"` for engaging summaries
- **Custom Personas**: Create your own analysis styles with `git-summary create-persona "Your Name"`

### For Developers
- **Architecture**: Review the PRD for system design and technical details
- **Contributing**: Check the main README for development setup instructions
- **Testing**: Run `uv run pytest` for the full test suite

## 🔧 Configuration Locations

- **Main config**: `~/.git-summary/config.json`
- **Custom personas**: `~/.git-summary/personas/*.yaml`
- **Environment variables**: `GITHUB_TOKEN`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`

## 📖 Example Workflows

### Daily Standups
```bash
git-summary ai-summary yourname --persona "Team Lead" --days 1
```

### Sprint Retrospectives
```bash
git-summary ai-summary team-repo --persona "Tech Analyst" --days 14
```

### Stakeholder Updates
```bash
git-summary ai-summary project --persona "Product Manager" --days 30
```

### Portfolio Showcase
```bash
git-summary ai-summary yourname --persona "Ghost Writer" --days 90
```

## 🆘 Need Help?

1. **Check the main [README](../README.md)** for basic troubleshooting
2. **Review persona documentation** in [personas.md](personas.md) for AI-related issues
3. **Open an issue** in the GitHub repository for bugs or feature requests

Happy analyzing! 🚀

---

## Technical Documentation Archive

### [PRD-GitHub-API-Integration.md](./PRD-GitHub-API-Integration.md)
**Product Requirements Document for GitHub API Integration**

Comprehensive specification for implementing the core GitHub API functionality that transforms git-summary from a CLI framework into a functional activity analyzer.

**Status:** ✅ Implemented
**Author:** Guilfoyle (Staff Engineer Review)
**Date:** 2025-01-13

## Architecture Overview

The project follows a modular architecture with clear separation of concerns:

```
src/git_summary/
├── cli.py              # Command-line interface
├── config.py           # Configuration management
├── github_client.py    # GitHub API client
├── fetchers.py         # API endpoint coordination
├── processors.py       # Event analysis and aggregation
└── models.py          # Data models and schemas
```

## Development Workflow

1. **Planning**: Technical requirements defined in PRDs
2. **Implementation**: Phased development following PRD milestones
3. **Testing**: Comprehensive unit, integration, and performance tests
4. **Documentation**: Living documentation updated with each release

## Contributing

All major features and architectural changes should have corresponding documentation in this directory before implementation begins.
