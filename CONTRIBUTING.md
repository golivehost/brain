# Contributing to GoLiveHost Brain

Thank you for your interest in contributing to GoLiveHost Brain! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)
- [Performance Considerations](#performance-considerations)
- [License](#license)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone. Please be considerate of differing viewpoints and experiences, and show courtesy to other community members.

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally.
3. Add the original repository as a remote named `upstream`.
4. Create a new branch for your feature or bug fix.
5. Make your changes and commit them with clear, descriptive messages.
6. Push your branch to your fork on GitHub.
7. Submit a pull request to the main repository.

## Development Environment

### Requirements

- PHP 8.0 or higher
- Composer

### Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/golivehost-brain.git
cd golivehost-brain

# Install dependencies
composer install

# Add upstream remote
git remote add upstream https://github.com/golivehost/brain.git
```

## Coding Standards

This project follows PSR-12 coding standards. Please ensure your code adheres to these standards.

### Key Guidelines

- Use 4 spaces for indentation, not tabs.
- Class names should be declared in `PascalCase`.
- Method names should be declared in `camelCase`.
- Constants should be declared in `UPPER_CASE`.
- Add appropriate PHPDoc comments for all classes, methods, and properties.
- Keep lines under 120 characters when possible.
- Use type hints for parameters and return types.

## Pull Request Process

1. Ensure your code follows the coding standards.
2. Update the README.md with details of changes if applicable.
3. Add or update tests as necessary.
4. Update documentation as needed.
5. Ensure compatibility with PHP 8.0 and above.
6. Include a clear description of the changes in your PR.

### PR Title Format

Use a clear, descriptive title for your PR that summarizes the changes. For example:

- "Add GRU implementation"
- "Fix memory leak in LSTM training"
- "Improve documentation for Matrix utilities"

## Testing

All new features and bug fixes should include tests. This project uses PHPUnit for testing.

### Running Tests

```bash
# Run all tests
composer run-script test

# Run specific test
./vendor/bin/phpunit --filter=TestName
```

### Writing Tests

- Place tests in the `tests` directory.
- Name test files with the suffix `Test.php` (e.g., `NeuralNetworkTest.php`).
- Test class names should match the file name.
- Each test method should begin with `test`.
- Write tests for both normal operation and edge cases.

## Documentation

Good documentation is crucial for this project. Please follow these guidelines:

- Update the README.md if you add or change functionality.
- Add PHPDoc comments to all classes, methods, and properties.
- Include examples for new features.
- Document parameters, return values, and exceptions.
- Keep documentation up-to-date with code changes.

## Issue Reporting

### Bug Reports

When reporting a bug, please include:

- A clear, descriptive title.
- Steps to reproduce the issue.
- Expected behavior.
- Actual behavior.
- PHP version and environment details.
- Code samples or test cases that demonstrate the issue.

### Feature Requests

For feature requests, please include:

- A clear, descriptive title.
- Detailed description of the proposed feature.
- Rationale for adding the feature.
- Examples of how the feature would be used.
- Any relevant references or resources.

## Performance Considerations

Neural network operations can be resource-intensive. When contributing, please consider:

- Memory usage
- Execution time
- Scalability with larger datasets
- Potential optimizations

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.

---

Thank you for contributing to GoLiveHost Brain!

Developed by: Go Live Web Solutions ([golive.host](https://golive.host))  
Author: Shubhdeep Singh ([GitHub](https://github.com/shubhdeepdev))
