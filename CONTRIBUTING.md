# Contributing to WazeCargo

## Workflow Overview

We use a branch-based workflow. All changes go through Pull Requests and require approval before merging to `main`.

```
main (protected)
  │
  └── feature/your-task-name (your working branch)
```

## Getting Started

### 1. Clone the repository (first time only)

```bash
git clone https://github.com/YOUR_USERNAME/WazeCargo.git
cd WazeCargo
```

### 2. Always start from an updated main

```bash
git checkout main
git pull
```

### 3. Create your feature branch

```bash
git checkout -b feature/your-task-name
```

Use this naming convention:

| Type | Pattern | Example |
|------|---------|---------|
| New feature | `feature/description` | `feature/directemar-scraper` |
| Bug fix | `fix/description` | `fix/scraper-timeout` |
| Documentation | `docs/description` | `docs/api-reference` |

## Making Changes

### 1. Make your changes and commit often

```bash
git add .
git commit -m "Add vessel arrival parsing"
```

Write clear commit messages:
- Start with a verb: Add, Fix, Update, Remove, Refactor
- Keep it short: under 50 characters
- Be specific: "Fix timeout in DIRECTEMAR scraper" not "Fix bug"

### 2. Push your branch to GitHub

```bash
git push -u origin feature/your-task-name
```

### 3. Create a Pull Request

1. Go to the repository on GitHub
2. Click "Compare & pull request" (or go to Pull Requests > New)
3. Fill in the template:
   - Title: Brief description of what you did
   - Description: What changed and why
   - Reference any related issues
4. Request review from the Tech Lead
5. Click "Create pull request"

### 4. Wait for review

- The Tech Lead will review your code
- If changes are requested, make them on your branch and push again
- Once approved, the Tech Lead will merge your PR

### 5. After your PR is merged

```bash
git checkout main
git pull
git branch -d feature/your-task-name
```

## Branch Protection Rules

The `main` branch is protected:

| Rule | Description |
|------|-------------|
| No direct pushes | All changes must go through Pull Requests |
| 1 approval required | Tech Lead must approve before merge |
| No force pushes | History cannot be rewritten |
| No deletions | Branch cannot be deleted |

## Task Assignments

| Team Member | Tasks |
|-------------|-------|
| Tech Lead | Pipeline, ML model, architecture, PR reviews |
| Junior 1 | DIRECTEMAR scraper, Valparaíso scraper, alert system |
| Junior 2 | San Antonio scraper, dashboard, training dataset |
| Junior 3 | VesselFinder scraper, EDA, baseline model |

## Code Standards

### Python

- Use descriptive variable names
- Add docstrings to functions
- Follow PEP 8 style guide
- Test your code locally before pushing

### Files

- Scrapers go in `scrapers/`
- Pipeline scripts go in `pipeline/`
- ML code goes in `ml/`
- Keep data in `data/raw/`

### Secrets

- Never commit credentials or API keys
- Use `.env` file locally (it's in `.gitignore`)
- For GitHub Actions, use repository Secrets

## Need Help?

1. Check the `docs/` folder
2. Ask in the team chat
3. Create an issue on GitHub

## Quick Reference

```bash
# Start new task
git checkout main
git pull
git checkout -b feature/my-task

# Save progress
git add .
git commit -m "Description of changes"
git push -u origin feature/my-task

# After PR merged
git checkout main
git pull
git branch -d feature/my-task
```
