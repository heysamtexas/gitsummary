# AI Personas Guide

Git-summary includes a powerful AI persona system that allows you to get different styles of analysis from the same GitHub activity data. Each persona has a unique voice, focus, and perspective on your development work.

## Overview

AI personas transform dry GitHub activity into engaging, insightful narratives. Whether you need technical depth, business impact analysis, or compelling stories for stakeholders, there's a persona that fits your needs.

## Built-in Personas

### Tech Analyst
**Focus:** Deep technical analysis
**Audience:** Engineers and technical teams
**Style:** Precise, technical, actionable insights
**Best For:** Code reviews, architecture discussions, engineering retrospectives

```bash
git-summary ai-summary username --persona "Tech Analyst"
```

### Product Manager
**Focus:** Business impact and user value
**Audience:** Stakeholders and cross-functional teams
**Style:** Business-focused, strategic insights
**Best For:** Sprint reviews, stakeholder updates, roadmap discussions

```bash
git-summary ai-summary username --persona "Product Manager"
```

### Ghost Writer
**Focus:** First-person developer narratives
**Audience:** Anyone who wants to understand the developer's journey
**Style:** Personal, engaging, relatable storytelling
**Best For:** Portfolio descriptions, blog posts, team retrospectives

```bash
git-summary ai-summary username --persona "Ghost Writer"
```

### Team Lead
**Focus:** Team collaboration and health
**Audience:** Team leads and management
**Style:** Supportive, growth-focused insights
**Best For:** Team retrospectives, performance discussions, mentoring

```bash
git-summary ai-summary username --persona "Team Lead"
```

### Data Analyst
**Focus:** Metrics and quantitative insights
**Audience:** Data-driven decision makers
**Style:** Statistical, analytical, KPI-focused
**Best For:** Performance tracking, trend analysis, reporting

```bash
git-summary ai-summary username --persona "Data Analyst"
```

## Custom Personas

You can create your own custom personas to match your specific needs, team culture, or communication style.

### Creating Custom Personas

Custom personas are stored in `~/.git-summary/personas/` and automatically loaded alongside built-in personas.

#### Quick Start

```bash
# Create a new custom persona
git-summary create-persona "My Custom Persona" --type basic

# Edit the YAML file to customize it
nano ~/.git-summary/personas/my_custom_persona.yaml

# Reload to pick up your changes
git-summary reload-personas

# Use your custom persona
git-summary ai-summary username --persona "My Custom Persona"
```

#### Template Types

**Basic Template:** General-purpose persona with flexible sections
```bash
git-summary create-persona "Marketing Writer" --type basic
```

**Technical Template:** Engineering-focused with technical depth
```bash
git-summary create-persona "Senior Architect" --type technical
```

### YAML Persona Format

Custom personas use a YAML configuration format that's easy to read and modify:

```yaml
name: "Your Persona Name"
description: "Brief description of what this persona does"
version: "1.0"
author: "your-name"

system_prompt: |
  You are a [role] who specializes in [focus area].

  Your role is to analyze GitHub activity from a [perspective] perspective, focusing on:
  - Key area 1
  - Key area 2
  - Key area 3

  Your communication style:
  - Tone guideline 1
  - Tone guideline 2
  - Target audience considerations

analysis_framework:
  sections:
    - name: "Section Name"
      description: "What this section should contain"
      max_length: 200
      format: "paragraph"  # or "bullet_list"
      max_items: 4  # for bullet lists
      optional: false  # true if section can be omitted

context_processing:
  commit_analysis:
    max_commits_displayed: 8
    include_line_changes: true
  pr_analysis:
    max_prs_displayed: 5
  issue_analysis:
    max_issues_displayed: 6

output_format:
  max_words: 400
  tone: "professional"
  audience: "general"
  include_metrics: true
```

### Configuration Options

#### System Prompt
The `system_prompt` defines your persona's personality, expertise, and communication style. This is where you specify:
- The persona's role and expertise
- Their analytical focus areas
- Communication style and tone
- Target audience considerations

#### Analysis Framework
The `analysis_framework` defines the structure of the output:

**Section Properties:**
- `name`: Section header
- `description`: What content should go in this section
- `max_length`: Maximum characters (for paragraphs)
- `format`: "paragraph" or "bullet_list"
- `max_items`: Maximum bullet points (for lists)
- `optional`: Whether section can be omitted if not relevant

#### Context Processing
Controls how GitHub data is processed and displayed:

```yaml
context_processing:
  commit_analysis:
    max_commits_displayed: 10    # How many commits to show
    include_line_changes: true   # Show +/- line counts
    categorize_by:               # How to group commits
      - "features"
      - "bug_fixes"
      - "refactoring"
  pr_analysis:
    max_prs_displayed: 5         # How many PRs to show
    include_review_comments: true
  issue_analysis:
    max_issues_displayed: 6      # How many issues to show
    focus_on_technical: false    # Prioritize technical issues
```

#### Output Format
Controls the final output style:

```yaml
output_format:
  max_words: 400              # Target word count
  tone: "conversational"      # professional, technical, conversational
  audience: "stakeholders"    # engineers, general, stakeholders
  include_metrics: true       # Show quantitative data
```

### Example Custom Personas

#### Marketing Communications Persona
```yaml
name: "Marketing Communicator"
description: "Transforms technical achievements into compelling marketing narratives"

system_prompt: |
  You are a Marketing Communications specialist who excels at translating
  technical achievements into compelling stories that resonate with diverse audiences.

  Focus on:
  - User impact and value creation
  - Innovation and competitive advantages
  - Growth and momentum indicators
  - Success stories and achievements

  Communication style:
  - Enthusiastic but authentic
  - Clear and jargon-free
  - Story-driven with concrete examples
  - Quantified impact when possible

analysis_framework:
  sections:
    - name: "Innovation Highlights"
      description: "Key technical achievements framed as innovations"
      format: "bullet_list"
      max_items: 3
    - name: "User Impact Story"
      description: "How this work improves user experience"
      max_length: 150
    - name: "Growth Indicators"
      description: "Metrics and progress signals"
      format: "bullet_list"
      max_items: 3

output_format:
  max_words: 300
  tone: "engaging"
  audience: "marketing_stakeholders"
```

#### Executive Summary Persona
```yaml
name: "Executive Briefer"
description: "Provides high-level strategic insights for executive audiences"

system_prompt: |
  You are an Executive Communications specialist who distills technical
  work into strategic business insights for senior leadership.

  Focus on:
  - Strategic impact and business value
  - Risk mitigation and competitive positioning
  - Resource allocation effectiveness
  - Future opportunities and challenges

  Communication style:
  - Concise and high-level
  - Business outcome focused
  - Data-driven when relevant
  - Forward-looking perspective

analysis_framework:
  sections:
    - name: "Strategic Summary"
      description: "High-level overview of business impact"
      max_length: 100
    - name: "Key Outcomes"
      description: "Measurable business results"
      format: "bullet_list"
      max_items: 3
    - name: "Future Considerations"
      description: "Strategic implications and next steps"
      max_length: 100

output_format:
  max_words: 250
  tone: "executive"
  audience: "senior_leadership"
```

## Managing Personas

### List Available Personas

```bash
# Standard view
git-summary personas

# Detailed view with sources
git-summary personas --sources
```

### Get Persona Information

```bash
git-summary persona-info "Persona Name"
```

### Reload Personas

After editing YAML files, reload to pick up changes:

```bash
git-summary reload-personas
```

### Persona Sources

Personas are loaded from three sources:

1. **Built-in:** Coded in Python (if any)
2. **Package:** YAML files included with git-summary
3. **Custom:** Your YAML files in `~/.git-summary/personas/`

Custom personas override package personas with the same name.

## Usage Examples

### Team Retrospective
```bash
# Get team-focused insights
git-summary ai-summary team-repo --persona "Team Lead" --days 14

# Get technical depth for engineering discussion
git-summary ai-summary team-repo --persona "Tech Analyst" --days 14
```

### Stakeholder Updates
```bash
# Business impact for product stakeholders
git-summary ai-summary project --persona "Product Manager" --days 30

# Executive summary for leadership
git-summary ai-summary project --persona "Executive Briefer" --days 30
```

### Portfolio and Blog Posts
```bash
# Personal narrative for portfolio
git-summary ai-summary yourname --persona "Ghost Writer" --days 90

# Technical showcase for engineering blog
git-summary ai-summary yourname --persona "Tech Analyst" --days 30
```

### Performance Analysis
```bash
# Quantitative insights for metrics review
git-summary ai-summary team --persona "Data Analyst" --days 60
```

## Best Practices

### Creating Effective Personas

1. **Define Clear Purpose:** What specific need does this persona address?
2. **Know Your Audience:** Who will read the analysis?
3. **Set Appropriate Tone:** Match the communication style to the context
4. **Balance Detail and Brevity:** Include enough depth without overwhelming
5. **Include Examples:** Test your persona with real data before sharing

### YAML Configuration Tips

1. **Use Clear Section Names:** Make headers self-explanatory
2. **Set Realistic Word Limits:** Consider your audience's attention span
3. **Test with Different Data:** Ensure the persona works with various activity levels
4. **Version Your Changes:** Keep track of persona evolution
5. **Document Customizations:** Add comments explaining your choices

### Team Sharing

1. **Create Team Repository:** Store shared personas in version control
2. **Establish Conventions:** Agree on naming and structure standards
3. **Regular Reviews:** Update personas based on feedback and changing needs
4. **Training Materials:** Help team members understand when to use which persona

## Troubleshooting

### Common Issues

**Persona Not Found**
```bash
# Check available personas
git-summary personas

# Reload if you just created/edited a persona
git-summary reload-personas
```

**YAML Syntax Errors**
```bash
# Check YAML syntax online or with tools
yamllint ~/.git-summary/personas/your-persona.yaml

# View detailed error information
git-summary reload-personas  # Will show specific errors
```

**Persona Not Loading**
- Ensure YAML file is in `~/.git-summary/personas/`
- Check file has `.yaml` or `.yml` extension
- Verify all required fields are present
- Run `git-summary reload-personas` to see error details

### Getting Help

For additional help with personas:
- Check the [GitHub repository](https://github.com/anthropics/claude-code) for examples
- Review built-in persona YAML files for reference
- Open an issue for bugs or feature requests

## Advanced Usage

### Persona Chaining
Use multiple personas for comprehensive analysis:

```bash
# Technical analysis
git-summary ai-summary project --persona "Tech Analyst" --output tech-analysis.md

# Business analysis
git-summary ai-summary project --persona "Product Manager" --output business-analysis.md

# Narrative summary
git-summary ai-summary project --persona "Ghost Writer" --output story.md
```

### Dynamic Persona Selection
Choose personas based on context:

```bash
# For sprint retrospectives
PERSONA="Team Lead"

# For stakeholder demos
PERSONA="Product Manager"

# For engineering discussions
PERSONA="Tech Analyst"

git-summary ai-summary project --persona "$PERSONA" --days 14
```

### Integration with CI/CD
Automate persona-based reporting:

```yaml
# GitHub Actions example
- name: Generate Weekly Summary
  run: |
    git-summary ai-summary ${{ github.repository_owner }} \
      --persona "Product Manager" \
      --days 7 \
      --output weekly-summary.md
```

The persona system makes git-summary incredibly flexible, allowing you to generate exactly the kind of analysis you need for any situation or audience.
