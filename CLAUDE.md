# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

Python project with virtual environment setup.

## Setup Instructions

### Environment Setup
1. Create virtual environment (already done): `python3 -m venv venv`
2. Activate virtual environment:
   - macOS/Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`
3. Install dependencies: `pip install -r requirements.txt`

### Development Commands
- **Run main script**: `python main.py`
- **Interactive REPL**: `python`
- **Deactivate venv**: `deactivate`

### Dependency Management
- **Install package**: `pip install <package-name>`
- **Save dependencies**: `pip freeze > requirements.txt`
- **Upgrade package**: `pip install --upgrade <package-name>`

### Testing Commands
- **Run tests**: (To be configured when test framework is added)
- **Coverage**: (To be configured when test framework is added)

### Lint/Format Commands
- **Linting**: (To be configured - consider using `ruff` or `flake8`)
- **Formatting**: (To be configured - consider using `black` or `ruff format`)

### Project Structure
```
processing/
├── venv/           # Virtual environment (git-ignored)
├── main.py         # Main application entry point
├── requirements.txt # Python dependencies
├── .gitignore      # Git ignore configuration
└── CLAUDE.md       # Project documentation for Claude Code
```

## Notes

- This file should be updated when the project structure is established
- Focus on project-specific commands and architecture, not generic best practices