# Contributing to twat-genai

Thank you for your interest in contributing to twat-genai! This document provides guidelines for contributors.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/twardoch/twat-genai.git
   cd twat-genai
   ```

2. **Set up development environment**:
   ```bash
   # Install UV (if not already installed)
   pip install uv
   
   # Create virtual environment
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install package in development mode
   uv pip install -e ".[all]"
   
   # Install development dependencies
   uv pip install pytest pytest-cov pytest-asyncio pytest-mock ruff mypy
   ```

## Development Scripts

We provide convenient scripts for common development tasks:

### Building
```bash
./scripts/build.sh
```

### Testing
```bash
./scripts/test.sh
```

### Releasing
```bash
./scripts/release.sh v1.0.0
```

## Code Quality

### Linting and Formatting
We use `ruff` for both linting and formatting:

```bash
# Check and fix linting issues
ruff check src/twat_genai tests --fix

# Format code
ruff format src/twat_genai tests
```

### Type Checking
We use `mypy` for type checking:

```bash
mypy src/twat_genai tests
```

### Testing
Run the test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run tests with coverage
python -m pytest --cov=src/twat_genai --cov-report=html tests/
```

## Git Workflow

1. **Fork the repository** on GitHub
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the code style guidelines
4. **Run tests** to ensure everything works
5. **Commit your changes** with descriptive messages
6. **Push to your fork** and create a pull request

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/) format:

- `feat: add new feature`
- `fix: resolve bug in component`
- `docs: update documentation`
- `test: add tests for feature`
- `refactor: improve code structure`
- `style: format code`
- `chore: update dependencies`

## Release Process

Releases are automated through GitHub Actions when you push a git tag:

1. **Ensure all tests pass**
2. **Update version** (handled automatically by git tags)
3. **Create and push tag**:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```
4. **GitHub Actions will**:
   - Run tests on multiple Python versions
   - Build multiplatform executables
   - Create GitHub release with binaries
   - Publish to PyPI

## Version Management

This project uses git-tag-based semantic versioning:

- **Major version** (v1.0.0): Breaking changes
- **Minor version** (v1.1.0): New features, backward compatible
- **Patch version** (v1.1.1): Bug fixes, backward compatible

Version is automatically determined from git tags using `hatch-vcs`.

## Testing Guidelines

- Write tests for new features
- Ensure good test coverage (aim for >80%)
- Use descriptive test names
- Mock external dependencies (FAL API calls)
- Test both success and failure cases

## Code Style

- Follow PEP 8 style guide
- Use type hints for all functions and methods
- Write clear docstrings
- Keep line length under 88 characters
- Use descriptive variable names

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions and classes
- Update INSTALLATION.md for setup changes
- Include examples in docstrings

## Pull Request Process

1. **Ensure all tests pass**
2. **Update documentation** if needed
3. **Add a clear description** of changes
4. **Link to any related issues**
5. **Wait for review** from maintainers
6. **Address feedback** if requested

## Questions?

Feel free to open an issue for questions or discussion about contributions.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).