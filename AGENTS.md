# Development Guidelines

This document contains critical information about working with this codebase. Follow these guidelines precisely.

## Environment Setup

To ensure everyone uses the same environment, follow these steps:

1. **Initial Setup**: Run `uv sync` to create/update your environment from the lockfile
2. **After Pulling Changes**: If `uv.lock` has changed, run `uv sync` again
3. **Adding Dependencies**: Use `uv add <package>` which updates both pyproject.toml and uv.lock
4. **Removing Dependencies**: Use `uv remove <package>`

The `uv.lock` file ensures all developers and CI/CD systems use exactly the same package versions.

### Running code

When running code always use the virtual environment based on `pyproject.toml` using uv. E.g., using `uv run`.

## Core Development Rules

1. Package Management
   - ONLY use uv, NEVER pip
   - Environment setup: `uv sync` (creates consistent environment from uv.lock)
   - Installation: `uv add package`
   - Running tools: `uv run tool`
   - Upgrading: `uv add --dev package --upgrade-package package`
   - FORBIDDEN: `uv pip install`, `@latest` syntax

2. Code Quality
   - Type hints required for all code
   - Public APIs must have docstrings
   - Functions must be focused and small
   - Follow existing patterns exactly
   - Line length: 88 chars maximum

3. Code Style
    - PEP 8 naming (snake_case for functions/variables)
    - Class names in PascalCase
    - Constants in UPPER_SNAKE_CASE
    - Document with docstrings
    - Use f-strings for formatting

4. Python specific styles
   - Follow the guidelines in [python-coding-guidelines.md](.augment/rules/python-coding-guidelines.md)
   - Prefer python generic type hints (e.g., list, dict) and explicit | None to the typing libraries List, Dict, Union, Optional
   - Create pythonic code using list/set/dict comprehensions and functional programming paradigms

5. Git commits
    - Always create a git commit after you have added any new functionality or completed a phase.
    - Follow the format defined in [commit-conventions.md](.augment/rules/commit-conventions.md)

## Development Philosophy

- **Simplicity**: Write simple, straightforward code
- **Readability**: Make code easy to understand
- **Performance**: Consider performance without sacrificing readability
- **Maintainability**: Write code that's easy to update
- **Testability**: Ensure code is testable
- **Reusability**: Create reusable components and functions
- **Less Code = Less Debt**: Minimize code footprint

## Coding Best Practices

- **Early Returns**: Use to avoid nested conditions
- **Descriptive Names**: Use clear variable/function names (prefix handlers with "handle")
- **Constants Over Functions**: Use constants where possible
- **DRY Code**: Don't repeat yourself
- **Functional Style**: Prefer functional, immutable approaches when not verbose
- **Minimal Changes**: Only modify code related to the task at hand
- **Function Ordering**: Define composing functions before their components
- **TODO Comments**: Mark issues in existing code with "TODO:" prefix
- **Simplicity**: Prioritize simplicity and readability over clever solutions
- **Build Iteratively** Start with minimal functionality and verify it works before adding complexity
- **Functional Code**: Use functional and stateless approaches where they improve clarity
- **Clean logic**: Keep core logic clean and push implementation details to the edges
- **File Organsiation**: Balance file organization with simplicity - use an appropriate number of files for the project scale


## Core Components

- `__main__.py`: Main entry point
- `api`: API for the project
- `tasks`: Tasks for the project
- `models`: Models for the project
- `loggers`: Loggers for the project
- `utils`: Utility functions for the project
- `tests`: Tests for the project
- `configs`: Configs for the project
- `data`: Data for the project


## Best Practices
   - Always create a git commit after you have added any new functionality or completed a phase.
   - Check git status before commits
   - Run formatters before type checks
   - Keep changes minimal
   - Document public APIs