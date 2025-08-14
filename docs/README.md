# Git-Summary Documentation

Welcome to the git-summary documentation! This directory contains comprehensive guides for using and extending git-summary.

## 📚 Available Documentation

### 🤖 AI Features
- **[AI Personas Guide](personas.md)** - Complete guide to AI personas, creating custom personas, and YAML configuration

### 📋 Technical Documentation
- **[PRD: GitHub API Integration](PRD-GitHub-API-Integration.md)** - Product Requirements Document for GitHub API integration
- **[Adaptive Analysis Architecture](#-adaptive-analysis-architecture)** - Deep dive into intelligent analysis strategies
- **[Performance Guide](#-performance--api-usage-guide)** - API usage patterns and optimization
- **[Troubleshooting Guide](#-troubleshooting-guide)** - Common issues and solutions

## 🚀 Quick Links

### For Users
- **Getting Started**: See the main [README](../README.md) for installation and basic usage
- **AI Analysis**: Use `git-summary ai-summary username --persona "Ghost Writer"` for engaging summaries with comprehensive engagement tracking
- **Repository Filtering**: Focus on specific projects with `git-summary ai-summary username --repo owner/project`
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
git-summary ai-summary yourname --persona "Team Lead" --days 1 --repo myorg/current-project
```

### Sprint Retrospectives
```bash
git-summary ai-summary yourname --persona "Tech Analyst" --days 14 --repo myorg/frontend --repo myorg/backend
```

### Stakeholder Updates
```bash
git-summary ai-summary yourname --persona "Product Manager" --days 30 --repo company/main-product
```

### Portfolio Showcase
```bash
git-summary ai-summary yourname --persona "Ghost Writer" --days 90
```

### Project-Specific Analysis
```bash
# Focus on a single repository
git-summary ai-summary yourname --repo company/main-app

# Compare activity across multiple repositories
git-summary ai-summary yourname --repo myorg/api --repo myorg/web --repo myorg/mobile

# Combine with time range for project sprints
git-summary ai-summary yourname --days 21 --repo team/sprint-project --persona "Data Analyst"
```

### Adaptive Analysis Examples (NEW)
```bash
# Automatic strategy selection for comprehensive coverage
git-summary summary kubernetes-maintainer --days 30

# Force comprehensive analysis for power users
git-summary summary high-activity-user --comprehensive

# Optimize for organizations with repository limits
git-summary summary enterprise-dev --max-repos 20

# Research mode with specific strategy
git-summary summary test-user --force-strategy multi_source
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

---

# 🧠 Adaptive Analysis Architecture

## Overview

The Adaptive Analysis system solves GitHub's 300-event API limitation through intelligent strategy selection and multi-source data discovery. Implemented across three core modules:

### Core Components

#### 1. AutomationDetector (`user_profiling.py`)
**Purpose**: Classify users to select optimal analysis strategy

**Algorithm**:
```python
def classify_user(username: str, days: int = 7) -> UserProfile:
    # Heuristic thresholds (tuned from real-world data)
    ISSUE_RATIO_THRESHOLD = 0.70        # 70%+ issue events = automation
    SINGLE_REPO_DOMINANCE = 0.80        # 80%+ single repo = bot
    HIGH_FREQUENCY_THRESHOLD = 5.0      # >5 events/day = automation
    
    # Multi-factor scoring for confidence
    confidence = weighted_average([
        issue_ratio_score,
        repo_dominance_score, 
        frequency_score,
        timing_pattern_score
    ])
    
    return UserProfile(is_automation=score > 0.7, confidence=confidence)
```

**Detection Logic**:
- **Issue Ratio Analysis**: Bots heavily use issue events for notifications
- **Repository Dominance**: Automation typically focuses on single repositories  
- **High Frequency Detection**: >5 events/day suggests automated activity
- **Timing Pattern Analysis**: Regular intervals indicate scripted behavior

#### 2. IntelligenceGuidedAnalyzer (`intelligence_guided.py`)
**Purpose**: Fast analysis for regular users (98% of cases)

**Three-Phase Approach**:
```python
async def discover_and_fetch(username: str, days: int):
    # Phase 1: Repository Discovery (300 events)
    events = await get_user_events_paginated(username, max_events=300)
    
    # Phase 2: Repository Scoring & Selection  
    repo_scores = self._score_repositories_by_development_activity(events)
    top_repos = sorted(repo_scores.items(), key=lambda x: x[1], reverse=True)[:15]
    
    # Phase 3: Deep-dive Analysis
    all_events = []
    for repo_name, score in top_repos:
        repo_events = await fetch_repository_events(repo_name, username, days)
        all_events.extend(repo_events)
        
    return deduplicate_events(all_events)
```

**Repository Scoring Algorithm**:
- **PushEvent**: 3 points (high development value)
- **PullRequestEvent**: 2 points (collaboration + code)  
- **ReleaseEvent**: 2 points (project milestones)
- **Other Events**: 1 point (general activity)
- **Minimum Threshold**: 2 points to filter noise

#### 3. MultiSourceDiscovery (`multi_source_discovery.py`)
**Purpose**: Comprehensive analysis for power users and automation

**Three-Source Strategy**:
```python
async def discover_and_fetch(username: str, days: int):
    # Source 1: Owned Repositories
    owned_repos = await github_client.get_user_repositories(username)
    
    # Source 2: Event-Based Discovery  
    event_repos = await self._discover_repositories_from_events(username)
    
    # Source 3: Commit Search Discovery
    search_repos = await self._discover_repositories_from_commits(username, days)
    
    # Merge with priority-based scoring
    merged_repos = self._merge_repository_sources(owned_repos, event_repos, search_repos)
    
    return await self._fetch_events_from_repositories(merged_repos, username, days)
```

**Priority-Based Repository Merging**:
1. **Base Contribution Score**: Activity level in repository  
2. **Multi-Source Bonus**: +50% if repository appears in multiple sources
3. **Owned Repository Boost**: 2x multiplier for user-owned repositories
4. **Final Ranking**: Top 25-50 repositories based on composite score

#### 4. AdaptiveRepositoryDiscovery (`adaptive_discovery.py`) 
**Purpose**: Main coordinator with production-grade reliability

**Strategy Selection Logic**:
```python
def _select_strategy(user_profile: UserProfile, force_strategy: str = None) -> str:
    if force_strategy:
        return validate_and_return(force_strategy)
        
    # Automatic selection based on user classification
    if user_profile.is_automation or user_profile.confidence > 0.8:
        return "multi_source"  # Comprehensive for power users/bots
    else:
        return "intelligence_guided"  # Fast for regular users
```

**Production Features**:
- **Circuit Breakers**: Graceful fallback when individual sources fail
- **Multi-Level Fallback**: Primary → Fallback → Emergency strategies
- **Event Deduplication**: Composite key deduplication across sources
- **Rate Limit Management**: Balances REST API vs Search API usage
- **Timeout Protection**: 5-minute analysis timeout with graceful cleanup

## API Usage Patterns

### Intelligence-Guided Strategy

**API Calls**: ~15-30 requests
**Endpoints Used**:
```
GET /users/{username}/events                    # 1-3 calls (300 events each)
GET /repos/{owner}/{repo}/events               # 5-15 calls (top repos)
GET /user (if authenticated)                   # 1 call (user validation)
```

**Rate Limiting**: 
- REST API: 5000 requests/hour (authenticated)
- Consumption: ~15-30 calls = 0.3-0.6% of quota
- Frequency: Can run ~150-300 analyses per hour

### Multi-Source Strategy

**API Calls**: ~60-120 requests  
**Endpoints Used**:
```
GET /users/{username}/repos                    # 1-3 calls (owned repos)
GET /users/{username}/events                   # 1-3 calls (event discovery)
GET /search/commits?q=author:{username}        # 3-10 calls (commit search)
GET /repos/{owner}/{repo}/events               # 30-80 calls (comprehensive)
GET /user (if authenticated)                   # 1 call (user validation)
```

**Rate Limiting**:
- REST API: ~90-110 calls = 1.8-2.2% of 5000/hour quota
- Search API: ~3-10 calls = 10-33% of 30/minute quota  
- **Limiting Factor**: Search API (30 requests/minute)
- Frequency: Can run ~6-10 analyses per hour (limited by Search API)

## Performance Characteristics

### Execution Time Analysis

**Intelligence-Guided** (Regular Users):
- **Network Time**: ~15-45 seconds (depends on API latency)
- **Processing Time**: ~5-15 seconds (scoring, deduplication)
- **Total Time**: ~30-60 seconds
- **Parallelization**: Repository fetches run concurrently

**Multi-Source** (Power Users):
- **Network Time**: ~90-240 seconds (more API calls)  
- **Processing Time**: ~15-45 seconds (complex merging, deduplication)
- **Total Time**: ~2-5 minutes
- **Parallelization**: All three sources discovered concurrently

### Memory Usage

**Data Structures**:
- **Events**: ~1-2KB per event object
- **Intelligence-Guided**: ~300-1500KB (300-750 events)
- **Multi-Source**: ~2-10MB (1000-5000 events)
- **Peak Memory**: ~50-100MB during processing (includes HTTP buffers)

**Optimization**:
- **Streaming**: Events processed as they arrive, not stored in memory
- **Deduplication**: Efficient set-based duplicate removal
- **Garbage Collection**: Explicit cleanup of large data structures

---

# 🚀 Performance & API Usage Guide

## Rate Limiting Strategy

### GitHub API Limits
- **REST API**: 5000 requests/hour (authenticated), 60/hour (unauthenticated)  
- **Search API**: 30 requests/minute (authenticated), 10/minute (unauthenticated)
- **Secondary Rate Limit**: ~1 request/second for bursts

### Optimization Techniques

#### 1. **Smart Endpoint Selection**
```python
# Efficient: Use specific repository endpoints
GET /repos/owner/repo/events         # Targeted data

# Inefficient: Over-fetch from user endpoints  
GET /users/username/events           # Generic data, needs filtering
```

#### 2. **Concurrent Request Processing**
```python
async def fetch_multiple_repositories(repos: List[str]):
    # Process repositories in parallel with rate limiting
    semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests
    
    async def fetch_with_limit(repo):
        async with semaphore:
            await rate_limiter.wait_if_needed()
            return await fetch_repository_events(repo)
    
    tasks = [fetch_with_limit(repo) for repo in repos]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

#### 3. **Adaptive Request Sizing**
- **Normal Users**: 100 events per page (GitHub maximum)
- **Power Users**: Smaller batches to avoid timeouts
- **Automation**: Custom page sizes based on activity patterns

## Usage Recommendations

### Daily Analysis Workflows
```bash
# Developers: Quick daily summaries
git-summary summary username --days 1
# Time: ~15-30 seconds, API calls: ~5-10

# Team leads: Weekly team analysis  
git-summary summary username --days 7 --max-repos 15
# Time: ~30-60 seconds, API calls: ~15-25
```

### Weekly/Monthly Reporting
```bash
# Individual contributors: Comprehensive monthly
git-summary summary username --days 30
# Auto-selects strategy, Time: ~1-3 minutes

# Open source maintainers: Full comprehensive analysis
git-summary summary username --days 30 --comprehensive  
# Time: ~3-5 minutes, API calls: ~80-120
```

### Organizational Usage
```bash
# Enterprise: Batch analysis with limits
for user in team_members; do
    git-summary summary $user --days 14 --max-repos 10
    sleep 30  # Rate limit pause between users
done
```

## Performance Optimization

### 1. **Repository Filtering**
```bash
# Focus analysis for faster results
git-summary summary username --max-repos 5     # 2-3x faster
git-summary summary username --days 7          # Reduce time scope
```

### 2. **Strategy Selection**
```bash
# Force fast strategy for quick insights
git-summary summary username --force-strategy intelligence_guided

# Only use comprehensive when needed  
git-summary summary username --comprehensive   # For power users only
```

### 3. **Timing Optimization**
- **Best Performance**: Off-peak hours (GitHub API less loaded)
- **Avoid**: Top of hour (many automated systems run then)
- **Batch Jobs**: Space analyses 30-60 seconds apart

---

# 🚨 Troubleshooting Guide

## Common Issues & Solutions

### Analysis Performance Issues

#### "Analysis timed out after 5 minutes"
**Cause**: User has extremely high activity or network latency issues
**Solutions**:
```bash
# Reduce scope
git-summary summary username --max-repos 10 --days 14

# Force faster strategy
git-summary summary username --force-strategy intelligence_guided

# Use smaller time windows
git-summary summary username --days 7
```

#### "API rate limit exceeded" 
**Cause**: Too many requests in short time period
**Solutions**:
```bash
# Check rate limit status
curl -H "Authorization: token YOUR_TOKEN" \
     -I https://api.github.com/user

# Wait for reset (shown in X-RateLimit-Reset header)
# Or use adaptive limits
git-summary summary username --max-repos 5
```

### Data Quality Issues

#### "Only found 50 events for active user"
**Cause**: User privacy settings or token permissions
**Solutions**:
1. **Check Token Scopes**: Ensure `public_repo` or `repo` scope
2. **User Privacy**: Some users limit public event visibility
3. **Force Comprehensive**: `git-summary summary username --comprehensive`

#### "Missing recent activity"
**Cause**: Event indexing delays or API caching
**Solutions**:
1. **Wait 10-15 minutes**: GitHub API has eventual consistency
2. **Check Multiple Sources**: Comprehensive analysis uses multiple endpoints
3. **Verify Repository Access**: Some events may be in private repositories

### Strategy Selection Issues

#### "System chose wrong strategy"
**Debugging**:
```bash
# Check user classification
git-summary summary username --force-strategy multi_source
# Compare with:
git-summary summary username --force-strategy intelligence_guided
```

**Manual Override**:
```bash
# Force specific strategy
git-summary summary username --comprehensive              # Multi-source
git-summary summary username --force-strategy intelligence_guided  # Fast mode
```

### Authentication Issues

#### "GitHub token is invalid"
**Causes**:
1. **Expired Token**: Regenerate at GitHub settings
2. **Insufficient Scopes**: Needs `public_repo` or `repo`
3. **Rate Limited Token**: Wait for reset

**Solutions**:
```bash
# Test token manually
curl -H "Authorization: token YOUR_TOKEN" https://api.github.com/user

# Reset authentication
git-summary auth-remove
git-summary auth
```

### Memory and Resource Issues

#### "Out of memory during analysis"
**Cause**: Very large analysis (>10,000 events)
**Solutions**:
```bash
# Limit scope
git-summary summary username --max-repos 15 --days 21

# Use streaming mode (reduce memory)  
git-summary summary username --max-events 2000
```

## Edge Cases & Limitations

### Private Repository Access
- **Limitation**: Can only analyze public repositories unless token has `repo` scope
- **Workaround**: Use Personal Access Token with `repo` scope for complete access
- **Detection**: Analysis will show warning for private repositories

### Deleted or Renamed Repositories
- **Behavior**: Events reference old repository names
- **Impact**: Some events may show as "unknown repository"
- **Mitigation**: Multi-source discovery helps identify current repository names

### Rate Limit Recovery
```bash
# Check current limits
curl -H "Authorization: token YOUR_TOKEN" \
     -I https://api.github.com/rate_limit

# Auto-recovery: System waits for rate limit reset
# Manual recovery: Wait or use different token
```

### Very New Accounts
- **Issue**: <7 days of history may not have enough data for automation detection
- **Behavior**: System defaults to intelligence-guided strategy  
- **Recommendation**: Use `--force-strategy multi_source` for complete analysis

## Performance Monitoring

### Built-in Diagnostics
```bash
# Analysis includes performance stats
git-summary summary username --output results.json
# Check "analysis_stats" section for timing and API usage
```

### API Usage Tracking
The system reports:
- **Total API Calls**: Number of requests made
- **Analysis Strategy**: Which approach was selected
- **Repository Count**: Number of repositories analyzed  
- **Event Count**: Total events discovered
- **Execution Time**: Wall-clock time for analysis

All major features and architectural changes should have corresponding documentation in this directory before implementation begins.
