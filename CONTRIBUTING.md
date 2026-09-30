# Contributing to PolarRoute

Thank you for considering contributing to PolarRoute!

## How to Contribute
- Fork the repository and create your branch from `main`.
- Ensure code is well-documented and type-annotated.
- Run tests and linters before submitting a pull request.
- Add or update documentation as needed.

## Development Setup

We provide a `Makefile` with common development commands to streamline your workflow. After cloning the repository, set up your development environment:

```sh
pip install -e . --group dev
```

### Makefile Commands

The following commands are available via the Makefile:

- **`make help`** - Show all available commands with descriptions
- **`make install`** - Install package with all dependencies
- **`make test`** - Run fast tests only (excludes slow tests)
- **`make test-all`** - Run all tests including slow ones
- **`make lint`** - Run linting checks with Ruff
- **`make format`** - Format code with Ruff
- **`make coverage`** - Generate coverage report (terminal and HTML)
- **`make clean`** - Clean build artifacts and cache files
- **`make docs`** - Build documentation
- **`make docs-clean`** - Clean documentation build artifacts

## Pre-commit Hooks
We use [pre-commit](https://pre-commit.com/) to automate code linting and formatting (Ruff). Please install and run pre-commit before pushing your changes:

```sh
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

This will ensure your code passes all style checks before submission.

## Pull Request Process
- Describe your changes in detail.
- Reference any related issues.
- Ensure all CI checks pass.

## Code Style
- Use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

## Reporting Issues
- Use the issue tracker to report bugs or request features.
- For bug reports, please include:
  - A clear and concise description of the bug and what you expected to happen.
  - Steps to reproduce the behaviour.
  - Screenshots and configuration/data files if relevant.
- For feature/enhancement requests, please include:
  - A clear description of the problem or feature you want.
  - The solution you'd like to see.
  - Tag the issue appropriately and check for duplicates before submitting.
