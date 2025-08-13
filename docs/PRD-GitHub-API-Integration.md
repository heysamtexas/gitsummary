# Product Requirements Document: GitHub API Integration

**Project**: git-summary CLI Tool
**Version**: 1.0
**Date**: 2025-01-13
**Author**: Guilfoyle (Staff Engineer Review)
**Status**: Approved for Implementation

## Executive Summary

Transform git-summary from a CLI framework into a functional GitHub activity analyzer by implementing comprehensive GitHub API integration. The system will fetch, process, and analyze user activity data to generate detailed summaries and insights.

## Codebase Analysis

### Strengths
- Clean CLI architecture with Typer and Rich
- Secure token management with proper file permissions
- Interactive flows for user experience
- Modern Python tooling (UV, Ruff, Pydantic)
- Good separation of concerns (auth, config, models)

### Current Gaps
- No GitHub API client implementation
- Models are placeholder stubs
- Missing data processing logic
- No rate limiting or pagination handling
- Summary command is a TODO placeholder

### Architectural Concerns
1. **Data Models**: Current models in `models.py` are empty - need full GitHub event schema
2. **API Client**: No HTTP client for GitHub API integration
3. **Data Processing**: No logic for aggregating and analyzing GitHub events
4. **Error Handling**: Minimal error handling for API failures
5. **Testing**: Basic auth tests only - need comprehensive API integration tests

## Functional Requirements

### FR1: GitHub API Client
**Priority: Critical**
- **FR1.1**: Implement authenticated HTTP client using stored/provided tokens
- **FR1.2**: Support multiple GitHub API endpoints with proper headers and authentication
- **FR1.3**: Handle API responses with proper error handling and status code validation
- **FR1.4**: Implement automatic retry logic for transient failures
- **FR1.5**: Support GitHub Enterprise endpoints (future enhancement)

### FR2: Data Fetching
**Priority: Critical**
- **FR2.1**: Fetch user events from `/user/events` endpoint (primary activity stream)
- **FR2.2**: Fetch user's public events from `/users/{username}/events/public`
- **FR2.3**: Fetch repository-specific events from `/repos/{owner}/{repo}/events`
- **FR2.4**: Support configurable date range filtering (default: 30 days)
- **FR2.5**: Handle pagination for large datasets automatically
- **FR2.6**: Implement intelligent endpoint selection based on data availability

### FR3: Event Processing and Analysis
**Priority: Critical**
- **FR3.1**: Parse and categorize GitHub events by type (PushEvent, IssuesEvent, PullRequestEvent, etc.)
- **FR3.2**: Generate daily activity rollups with event counts and repository breakdown
- **FR3.3**: Calculate summary statistics (total events, active repositories, event distribution)
- **FR3.4**: Group events by repository for repository-level analysis
- **FR3.5**: Extract meaningful metadata from event payloads (commit messages, PR titles, etc.)

### FR4: Output Generation
**Priority: Critical**
- **FR4.1**: Generate structured JSON output matching defined schema
- **FR4.2**: Support console output with Rich formatting for human readability
- **FR4.3**: Provide summary view and detailed view options
- **FR4.4**: Export to file with configurable output path
- **FR4.5**: Support JSON streaming for large datasets

### FR5: Rate Limiting and Performance
**Priority: High**
- **FR5.1**: Implement GitHub API rate limiting compliance (5000/hour for authenticated requests)
- **FR5.2**: Display rate limit status and remaining quota to users
- **FR5.3**: Implement intelligent backoff when approaching rate limits
- **FR5.4**: Cache frequently accessed data to reduce API calls
- **FR5.5**: Optimize API calls by using appropriate endpoint combinations

### FR6: Error Handling and Resilience
**Priority: High**
- **FR6.1**: Handle authentication errors with clear messaging
- **FR6.2**: Gracefully handle network timeouts and connection errors
- **FR6.3**: Provide meaningful error messages for API failures
- **FR6.4**: Support partial data recovery when some endpoints fail
- **FR6.5**: Implement circuit breaker pattern for repeated failures

### FR7: CLI Integration
**Priority: Medium**
- **FR7.1**: Integrate with existing interactive username prompts
- **FR7.2**: Support existing filtering placeholders (--include-events, --exclude-repos)
- **FR7.3**: Add progress indicators for long-running API calls
- **FR7.4**: Provide verbose mode for debugging API interactions
- **FR7.5**: Support dry-run mode to preview API calls without execution

## Technical Architecture

### Core Components

#### 1. GitHub API Client (`github_client.py`)
```python
class GitHubClient:
    - Base HTTP client with authentication
    - Rate limiting enforcement
    - Automatic pagination handling
    - Error handling and retries
    - Response caching layer
```

#### 2. Event Fetchers (`fetchers.py`)
```python
class EventFetcher:
    - UserEventsFetcher: /user/events
    - PublicEventsFetcher: /users/{username}/events/public
    - RepoEventsFetcher: /repos/{owner}/{repo}/events
    - Parallel fetching coordination
```

#### 3. Data Processors (`processors.py`)
```python
class EventProcessor:
    - Event type classification
    - Time-based aggregation
    - Repository grouping
    - Summary statistics calculation
```

#### 4. Enhanced Models (`models.py`)
```python
# Complete GitHub event models
GitHubEvent, PushEvent, IssuesEvent, PullRequestEvent
ActivitySummary, DailyRollup, RepositoryBreakdown
```

### Data Flow Architecture

```
CLI Command → Authentication → GitHub API Client → Event Fetchers →
Data Processors → Output Generators → File/Console Output
```

### GitHub API Endpoints Integration

#### Primary Endpoints
1. **`/user/events`** - Authenticated user's activity (private repos included)
2. **`/users/{username}/events/public`** - Public activity for any user
3. **`/repos/{owner}/{repo}/events`** - Repository-specific events

#### Endpoint Selection Strategy
- Authenticated user querying own data: Use `/user/events` (most comprehensive)
- Public user analysis: Use `/users/{username}/events/public`
- Repository focus: Combine user events with `/repos/{owner}/{repo}/events`

### JSON Schema Implementation

```json
{
  "user": "string",
  "period": {
    "start": "ISO8601 datetime",
    "end": "ISO8601 datetime"
  },
  "summary": {
    "total_events": "integer",
    "repositories_active": "integer",
    "event_breakdown": {
      "PushEvent": "integer",
      "IssuesEvent": "integer",
      "PullRequestEvent": "integer"
    }
  },
  "daily_rollups": [
    {
      "date": "ISO8601 date",
      "events": "integer",
      "repositories": ["string"]
    }
  ],
  "repository_breakdown": {
    "repo_name": {
      "events": "integer",
      "event_types": {},
      "last_activity": "ISO8601 datetime"
    }
  },
  "detailed_events": [
    {
      "type": "string",
      "created_at": "ISO8601 datetime",
      "repository": "string",
      "details": {}
    }
  ]
}
```

## Error Handling Strategy

### Error Categories and Responses

#### Authentication Errors (401/403)
- Clear message about token validity
- Redirect to `git-summary auth` command
- Support for token refresh workflow

#### Rate Limiting (403 with rate limit headers)
- Display current rate limit status
- Estimated wait time until reset
- Option to continue with reduced functionality

#### Network Errors
- Automatic retry with exponential backoff
- Graceful degradation when some endpoints fail
- Progress preservation for long-running operations

#### Data Processing Errors
- Partial results when possible
- Detailed logging of problematic events
- Continue processing remaining data

## Performance Considerations

### Rate Limiting Strategy
```python
# Implement intelligent rate limiting
class RateLimiter:
    - Track remaining requests across endpoints
    - Implement exponential backoff
    - Priority-based request queuing
    - Rate limit status reporting
```

### Pagination Optimization
- Parallel fetching where possible
- Smart page size selection based on data volume
- Progress tracking for large datasets
- Resume capability for interrupted operations

### Caching Layer
- In-memory cache for session-based requests
- Optional file-based cache for expensive operations
- Cache invalidation based on data freshness requirements
- Configurable cache TTL per endpoint

## Implementation Phases

### Phase 1: Core API Integration (Week 1-2)
**Deliverables:**
- GitHub API client with authentication
- Basic event fetching from `/user/events`
- Simple JSON output generation
- Rate limiting implementation

**Acceptance Criteria:**
- Successfully authenticate and fetch user events
- Handle rate limiting gracefully
- Generate basic JSON output matching schema

### Phase 2: Comprehensive Event Processing (Week 2-3)
**Deliverables:**
- All major GitHub event types supported
- Daily rollup calculations
- Repository breakdown analysis
- Rich console output formatting

**Acceptance Criteria:**
- Process all GitHub event types correctly
- Generate comprehensive daily and repository summaries
- Beautiful console output with Rich

### Phase 3: Advanced Features and Polish (Week 3-4)
**Deliverables:**
- Multiple endpoint integration
- Filtering implementation (--include-events, --exclude-repos)
- Progress indicators and verbose mode
- Comprehensive error handling

**Acceptance Criteria:**
- Support filtering by event types and repositories
- Robust error handling for all failure modes
- Professional user experience with progress feedback

### Phase 4: Optimization and Testing (Week 4)
**Deliverables:**
- Performance optimization
- Comprehensive test suite
- Documentation and examples
- CI/CD pipeline integration

**Acceptance Criteria:**
- Handle large datasets efficiently
- 90%+ test coverage
- Complete documentation
- Automated testing in CI

## Testing Strategy

### Unit Tests
- GitHub API client methods
- Event processing functions
- Data model validation
- Configuration management

### Integration Tests
- End-to-end API workflows
- Authentication flows
- Error handling scenarios
- Output generation validation

### Performance Tests
- Large dataset processing
- Rate limiting behavior
- Memory usage optimization
- Concurrent request handling

### Manual Testing Scenarios
- Interactive authentication flows
- Various GitHub user profiles
- Network failure conditions
- Rate limiting edge cases

## Success Metrics

### Functional Metrics
- Successfully process 99%+ of GitHub event types
- Handle datasets up to 10,000 events without memory issues
- Complete analysis within 30 seconds for typical 30-day periods
- Maintain <1% error rate for API interactions

### User Experience Metrics
- Interactive flows complete in <5 steps
- Clear error messages with actionable guidance
- Progress feedback for operations >5 seconds
- Support for 100% of planned CLI options

## Risk Assessment

### High Risk Items
1. **GitHub API Changes**: Mitigation through versioned API usage and monitoring
2. **Rate Limiting**: Mitigation through intelligent request management
3. **Large Dataset Performance**: Mitigation through streaming and pagination optimization

### Medium Risk Items
1. **Authentication Complexity**: Mitigation through comprehensive testing
2. **Event Schema Changes**: Mitigation through flexible parsing logic

### Low Risk Items
1. **CLI Integration**: Well-established patterns
2. **Output Generation**: Standard JSON serialization

## Architecture Recommendations

Based on codebase analysis, here are critical architectural improvements:

### 1. Separate Concerns Properly
- Move GitHub API logic into dedicated client module
- Keep data processing separate from API fetching
- Isolate output formatting from business logic

### 2. Implement Proper Error Boundaries
- API errors should not crash the entire application
- Partial success scenarios should still produce useful output
- Clear error propagation with context

### 3. Design for Testability
- Inject HTTP client for testing
- Mock GitHub API responses
- Separate pure functions from side effects

### 4. Consider Data Volume Early
- GitHub power users can have thousands of events
- Memory-efficient processing for large datasets
- Streaming JSON output for very large results

## Implementation Priority

### Critical Files to Create/Update
1. `src/git_summary/fetchers.py` - API endpoint coordination
2. `src/git_summary/processors.py` - Event analysis and aggregation
3. `src/git_summary/models.py` - Complete GitHub event schemas (major expansion)
4. `src/git_summary/github_client.py` - Production-ready API client (enhance existing)
5. `src/git_summary/cli.py` - Wire everything together in summary command (update)

### Implementation Order
1. **Enhanced API Client** - Authentication, rate limiting, error handling
2. **Event Models** - Complete Pydantic models for all GitHub event types
3. **Event Fetchers** - Coordinate API calls across multiple endpoints
4. **Data Processors** - Aggregate and analyze fetched events
5. **CLI Integration** - Connect all components in summary command
6. **Output Generation** - Format and export processed data

This PRD provides a comprehensive roadmap for implementing robust GitHub API integration while maintaining the excellent foundation that exists. The phased approach ensures incremental value delivery while building toward full functionality.

**Next Steps**: Begin Phase 1 implementation with enhanced GitHub API client and basic event fetching capabilities.
