# Design Document: Git Publication Preparation

## Overview

This design outlines the approach for preparing the СберИндекс Anomaly Detection System for public Git repository publication. The system will be cleaned of sensitive data, properly configured with Git settings, and documented for open-source distribution.

## Architecture

### Component Overview

```
Git Publication Preparation
├── Data Protection Layer
│   ├── .gitignore Configuration
│   ├── Sensitive File Removal
│   └── Data Documentation
├── Documentation Layer
│   ├── README Enhancement
│   ├── License Addition
│   └── Contribution Guidelines
├── Repository Configuration
│   ├── .gitattributes Setup
│   └── Git History Cleanup
└── Validation Layer
    └── Pre-publication Checks
```

## Components and Interfaces

### 1. Data Protection Component

**Purpose**: Ensure no sensitive data is committed to the public repository

**Implementation**:

#### .gitignore File Structure

```gitignore
# Data files - NEVER commit
*.parquet
*.xlsx
*.gpkg
*.csv
connection.parquet
consumption.parquet
market_access.parquet
rosstat/
t_dict_municipal/

# Output directory
output/
*.log

# Python cache
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.pytest_cache/
*.cover
.coverage
htmlcov/

# Virtual environments
.env
.venv
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Temporary files
*.tmp
*.bak
*.swp
AUDIT_SUMMARY.md
CHANGES_APPLIED.md
PROJECT_READINESS_ASSESSMENT.md
RESULTS_ASSESSMENT.md
```

#### Sensitive File Identification

Files to evaluate for removal:
- Temporary documentation (AUDIT_SUMMARY.md, CHANGES_APPLIED.md, etc.)
- Implementation summaries in docs/ (task_*.md) - keep only essential ones
- Example output files if they contain real data

### 2. Documentation Enhancement Component

**Purpose**: Provide clear, comprehensive documentation for users and contributors

#### README.md Improvements

Current README is comprehensive (1000+ lines). Enhancements needed:
- Add clear "Getting Started" section at the top
- Add badges (Python version, license, etc.)
- Simplify initial sections for quick understanding
- Add "Data Requirements" section explaining where to get data
- Add "Contributing" section or link to CONTRIBUTING.md
- Add "License" section with clear license type

#### New Documentation Files

**LICENSE**:
- Choose appropriate open-source license (MIT, Apache 2.0, GPL, etc.)
- Add full license text to root directory
- Reference in README.md

**CONTRIBUTING.md** (optional but recommended):
```markdown
# Contributing Guidelines

## Code Style
- Follow PEP 8 for Python code
- Use black for formatting
- Run flake8 for linting

## Testing
- Write tests for new features
- Ensure all tests pass: `pytest`
- Maintain test coverage above 80%

## Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit pull request with clear description
```

**DATA_GUIDE.md** (new file):
```markdown
# Data Requirements Guide

## Required Data Files

This project requires the following data files to run:

### СберИндекс Data (Parquet format)
- `connection.parquet` - Territorial connections (4.7M records)
- `consumption.parquet` - Consumption data
- `market_access.parquet` - Market access metrics

### Rosstat Data (Parquet format)
Place in `rosstat/` directory:
- `2_bdmo_population.parquet` - Population statistics
- `3_bdmo_migration.parquet` - Migration data
- `4_bdmo_salary.parquet` - Salary information

### Municipal Dictionary (Excel format)
Place in `t_dict_municipal/` directory:
- `t_dict_municipal_districts.xlsx` - Municipal metadata

## Where to Get Data

[Instructions for obtaining data from official sources]

## Data Privacy

All data files are excluded from Git via .gitignore. Never commit actual data files.
```

### 3. Repository Configuration Component

**Purpose**: Configure Git settings for cross-platform compatibility and proper file handling

#### .gitattributes File

```gitattributes
# Auto detect text files and perform LF normalization
* text=auto

# Python files
*.py text eol=lf
*.pyi text eol=lf

# Configuration files
*.yaml text eol=lf
*.yml text eol=lf
*.json text eol=lf
*.md text eol=lf
*.txt text eol=lf

# Shell scripts
*.sh text eol=lf

# Windows scripts
*.bat text eol=crlf
*.cmd text eol=crlf
*.ps1 text eol=crlf

# Binary files
*.parquet binary
*.xlsx binary
*.gpkg binary
*.png binary
*.jpg binary
*.jpeg binary
```

### 4. Git History Cleanup Component

**Purpose**: Remove sensitive files from Git history if previously committed

**Approach**:
- Check if sensitive files exist in Git history
- If found, use `git filter-branch` or `BFG Repo-Cleaner` to remove
- Force push to clean repository (only if not yet public)

**Commands**:
```bash
# Check for sensitive files in history
git log --all --full-history -- "*.parquet"

# If found, remove using BFG (recommended)
bfg --delete-files "*.parquet"
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

## Data Models

### File Structure After Cleanup

```
sberindex-anomaly-detection/
├── .git/
├── .gitignore              # NEW/UPDATED
├── .gitattributes          # NEW
├── LICENSE                 # NEW
├── README.md               # UPDATED
├── CONTRIBUTING.md         # NEW (optional)
├── DATA_GUIDE.md           # NEW
├── requirements.txt
├── config.yaml
├── main.py
├── src/
│   └── [all source files]
├── tests/
│   └── [all test files]
├── examples/
│   └── [example scripts]
├── docs/
│   ├── [essential docs only]
│   └── [remove task_*.md if too verbose]
└── .kiro/
    └── [keep for development]
```

### Files to Remove/Archive

**Remove from repository**:
- AUDIT_SUMMARY.md
- CHANGES_APPLIED.md
- PROJECT_READINESS_ASSESSMENT.md
- RESULTS_ASSESSMENT.md
- QUICK_START.md (content merged into README)

**Evaluate for removal**:
- docs/task_*.md files (25+ files) - keep only if valuable for users
- Alternatively, move to docs/implementation/ subdirectory

## Error Handling

### Pre-publication Validation

**Checklist**:
1. ✓ No .parquet files in repository
2. ✓ No .xlsx files in repository
3. ✓ No output/ directory committed
4. ✓ .gitignore properly configured
5. ✓ LICENSE file present
6. ✓ README.md has license reference
7. ✓ No API keys or credentials in code
8. ✓ All paths are relative (no absolute paths)
9. ✓ requirements.txt is up to date
10. ✓ Tests pass successfully

**Validation Script** (to be created):
```python
# validate_for_publication.py
import os
import subprocess

def check_sensitive_files():
    """Check for sensitive files in repository"""
    sensitive_patterns = ['*.parquet', '*.xlsx', 'output/']
    # Implementation
    
def check_required_files():
    """Check for required files"""
    required = ['.gitignore', 'LICENSE', 'README.md', 'requirements.txt']
    # Implementation
    
def run_tests():
    """Run test suite"""
    result = subprocess.run(['pytest'], capture_output=True)
    return result.returncode == 0
```

## Testing Strategy

### Pre-publication Tests

1. **Clean Clone Test**:
   - Clone repository to fresh directory
   - Verify no data files present
   - Verify installation works: `pip install -r requirements.txt`
   - Verify tests pass: `pytest`

2. **Documentation Test**:
   - Follow README instructions from scratch
   - Verify all links work
   - Verify examples run (with sample data)

3. **Cross-platform Test**:
   - Test on Windows, Linux, macOS
   - Verify line endings are correct
   - Verify paths work on all platforms

### Test Checklist

```markdown
- [ ] Fresh clone installs successfully
- [ ] No sensitive data files present
- [ ] All tests pass
- [ ] README instructions are clear
- [ ] License is properly specified
- [ ] .gitignore excludes all sensitive files
- [ ] .gitattributes handles line endings
- [ ] No absolute paths in code
- [ ] requirements.txt is complete
- [ ] Examples run without errors (with sample data)
```

## Implementation Notes

### License Selection Guidance

**Recommended licenses**:

1. **MIT License** (most permissive):
   - Allows commercial use
   - Minimal restrictions
   - Good for maximum adoption

2. **Apache 2.0** (patent protection):
   - Similar to MIT
   - Includes patent grant
   - Good for projects with potential patents

3. **GPL v3** (copyleft):
   - Requires derivatives to be open source
   - Good for ensuring openness

**Recommendation**: MIT License for maximum flexibility and adoption

### README Structure Recommendation

```markdown
# Project Title

[Badges: Python version, License, etc.]

## Quick Start

[3-5 lines: what it does, how to install, how to run]

## Features

[Bullet points of key features]

## Installation

[Step-by-step installation]

## Usage

[Basic usage examples]

## Data Requirements

[Link to DATA_GUIDE.md]

## Documentation

[Links to detailed docs]

## Contributing

[Link to CONTRIBUTING.md or inline guidelines]

## License

[License type and link to LICENSE file]

## Acknowledgments

[Credits and thanks]
```

### Temporary Files Handling

**Strategy**:
- Move temporary docs to a `docs/archive/` directory
- Add `docs/archive/` to .gitignore
- Keep only essential documentation in main docs/

### Sample Data Consideration

**Option 1**: No sample data
- Provide clear instructions on obtaining data
- Document expected data format

**Option 2**: Synthetic sample data
- Create small synthetic dataset for testing
- Include in `examples/sample_data/`
- Document that it's synthetic

**Recommendation**: Option 1 (no sample data) to avoid confusion

## Design Decisions

### Why .gitignore is Critical

- Prevents accidental commits of sensitive data
- Reduces repository size
- Protects user privacy
- Ensures clean repository

### Why .gitattributes Matters

- Ensures consistent line endings across platforms
- Prevents merge conflicts due to line ending differences
- Properly handles binary files
- Critical for Windows/Linux/Mac compatibility

### Documentation Philosophy

- README should be comprehensive but scannable
- Separate detailed docs into dedicated files
- Provide quick start for impatient users
- Provide deep dive for thorough users

### License Choice Impact

- Affects how others can use the code
- Affects commercial viability
- Affects contribution requirements
- Should align with project goals

## Security Considerations

### Data Privacy

- Never commit actual data files
- Ensure .gitignore is comprehensive
- Check Git history for leaked data
- Document data sources without including data

### Credential Management

- No API keys in code
- No passwords in configuration
- Use environment variables for secrets
- Document secret management approach

### Path Security

- Use relative paths only
- No hardcoded user directories
- No absolute paths that reveal system structure

## Performance Considerations

### Repository Size

- Excluding data files keeps repo small
- Small repo = faster clones
- Faster CI/CD pipelines
- Better developer experience

### Documentation Size

- Balance between comprehensive and overwhelming
- Consider moving verbose docs to wiki or separate repo
- Keep essential docs in main repo

## Future Enhancements

### Potential Additions

1. **GitHub Actions CI/CD**:
   - Automated testing on push
   - Automated linting
   - Automated documentation generation

2. **Docker Support**:
   - Dockerfile for easy deployment
   - Docker Compose for full stack

3. **Sample Data Generator**:
   - Script to generate synthetic data
   - Helps users test without real data

4. **Wiki or GitHub Pages**:
   - Detailed documentation
   - Tutorials and guides
   - API documentation

### Maintenance Plan

- Regular dependency updates
- Security vulnerability scanning
- Documentation updates
- Community engagement

## Conclusion

This design provides a comprehensive approach to preparing the project for Git publication. The focus is on:
1. Protecting sensitive data
2. Providing clear documentation
3. Ensuring cross-platform compatibility
4. Following open-source best practices

Implementation will be done incrementally, with validation at each step to ensure nothing is broken and no sensitive data is exposed.
