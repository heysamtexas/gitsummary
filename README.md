# GitHub Activity Tracker

A comprehensive tool for querying the GitHub API to extract, analyze, and summarize user activity across repositories and projects.

## Overview

This project creates an automated system that connects to GitHub's API using OAuth2 or Personal Access Tokens to retrieve comprehensive user activity data. The system processes commits, comments, issues, pull requests, and other events to generate structured summaries and daily rollups.

## Core Functionality

### Data Collection
- **Authentication**: Support for both OAuth2 tokens and Personal Access Tokens
- **User Activity Extraction**: Pull comprehensive event data for a specified user
- **Event Types Covered**:
  - Commits and commit comments
  - Issues (created, commented on, assigned, closed)
  - Pull requests (created, reviewed, merged, commented)
  - Repository events (stars, forks, watches)
  - Project board activities
  - Code reviews and review comments
  - Release activities

### Data Processing & Analysis
- **Time-based Summaries**: Generate summaries for daily, weekly, or custom time ranges
- **Repository Grouping**: Organize activities by repository or project
- **Event Count Analysis**: Summarize activities by event type and frequency
- **Comprehensive Data Extraction**: Ensure no user activity is missed

### Output Generation
- **Daily Rollups**: Automated daily summaries of all user activities
- **JSON Export**: Structured JSON output for downstream consumption
- **Flexible Filtering**: Support for date ranges, repository filters, and event type filters

## Key Features

### 1. Comprehensive Activity Tracking
- Monitor all user interactions across GitHub
- Track both direct actions and collaborative activities
- Include timeline information for all events

### 2. Flexible Summarization
- **Daily summaries**: Complete rollup of daily activities
- **Time-based queries**: Custom date range analysis
- **Event-count summaries**: Activity frequency analysis
- **Repository breakdown**: Per-repo activity analysis

### 3. Structured Output
- Clean JSON format for easy integration
- Hierarchical data organization (user → repo → events)
- Standardized event schema across all activity types
- Metadata inclusion (timestamps, event types, related objects)

## API Requirements

### GitHub API Endpoints
The system will need to query multiple GitHub API endpoints:
- `/user/events` - User's public events
- `/user/events/orgs/{org}` - Organization events
- `/repos/{owner}/{repo}/events` - Repository-specific events
- `/user/received_events` - Events received by user
- `/repos/{owner}/{repo}/commits` - Repository commits
- `/repos/{owner}/{repo}/issues` - Issues and pull requests
- `/repos/{owner}/{repo}/pulls` - Pull request details
- `/user/subscriptions` - Watched repositories

### Authentication
- Support for GitHub Personal Access Tokens
- OAuth2 flow implementation for secure token management
- Token validation and scope verification

## Data Structure

### Input Parameters
- **User identifier**: GitHub username or authenticated user
- **Time range**: Start and end dates for activity query
- **Repository filters**: Optional list of specific repositories
- **Event type filters**: Optional filtering by activity type
- **Output format preferences**: Granularity and grouping options

### Output Schema
```json
{
  "user": "username",
  "period": {
    "start": "ISO8601 timestamp",
    "end": "ISO8601 timestamp"
  },
  "summary": {
    "total_events": 0,
    "repositories_active": 0,
    "event_breakdown": {}
  },
  "daily_rollups": [],
  "repository_breakdown": {},
  "detailed_events": []
}
```

## Technical Requirements

### Core Dependencies
- GitHub API client library
- JSON processing capabilities
- Date/time handling for time-based queries
- HTTP client for API requests
- Authentication token management

### Performance Considerations
- Rate limiting compliance with GitHub API limits
- Efficient pagination handling for large datasets
- Caching mechanisms for frequently accessed data
- Batch processing for multiple repositories

### Error Handling
- Token validation and expiration handling
- API rate limit management
- Network error recovery
- Data validation and sanitization

## Use Cases

### Personal Productivity Tracking
- Daily activity summaries for personal review
- Contribution tracking across multiple projects
- Time-based productivity analysis

### Team Management
- Individual contributor activity monitoring
- Project participation tracking
- Cross-repository collaboration analysis

### Project Analytics
- Repository activity trends
- Contributor engagement metrics
- Development velocity tracking

## Integration Points

### Upstream Consumption
- Clean JSON output format for easy parsing
- Standardized data schema across all summaries
- Webhook-ready format for real-time integrations
- Support for various aggregation levels

### Extensibility
- Modular design for adding new GitHub event types
- Configurable output formats
- Plugin architecture for custom summarization logic
- API endpoint flexibility for GitHub Enterprise

## Configuration Options

### Time-based Queries
- Daily rollups (default)
- Custom date ranges
- Rolling window summaries (last 7 days, 30 days, etc.)
- Real-time vs batch processing modes

### Output Customization
- Verbosity levels (summary only vs detailed events)
- Repository grouping preferences
- Event type inclusion/exclusion filters
- Time zone handling for global teams

## Security & Privacy

### Token Management
- Secure storage of authentication tokens
- Minimal required scope permissions
- Token rotation and refresh handling
- Audit logging for API access

### Data Privacy
- User consent for data collection
- Data retention policies
- Export/deletion capabilities
- Compliance with data protection regulations

## Success Metrics

The system should successfully:
1. Authenticate with GitHub API using provided tokens
2. Extract comprehensive user activity data
3. Generate accurate daily rollups without missing events
4. Output clean, structured JSON for downstream consumption
5. Handle API rate limits gracefully
6. Process data efficiently for users with high activity volumes

## Future Enhancements

- Real-time webhook integration
- Multi-user batch processing
- Advanced analytics and trend analysis
- Integration with other development tools
- Custom notification systems
- Historical data analysis capabilities
