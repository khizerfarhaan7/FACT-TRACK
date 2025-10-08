# Contributing to FACT-TRACK

Thank you for your interest in contributing to FACT-TRACK! This document provides guidelines and instructions for contributing to this project.

## 🔒 Security Notice

**IMPORTANT**: This repository is protected by strict security measures. All changes must go through pull requests and require approval from the repository owner.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Development Guidelines](#development-guidelines)
- [Security Guidelines](#security-guidelines)
- [Reporting Issues](#reporting-issues)

## 🤝 Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- Be respectful and inclusive
- Provide constructive feedback
- Follow security best practices
- Respect intellectual property rights
- Maintain professional communication

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of machine learning and NLP
- Familiarity with PyTorch and Transformers

### Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/FACT-TRACK.git
   cd FACT-TRACK
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🔄 How to Contribute

### 1. Create an Issue First

Before making any changes:
1. Check existing issues to avoid duplicates
2. Create a new issue describing your proposed change
3. Wait for approval from the repository owner
4. Get assigned to the issue

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number
```

### 3. Make Your Changes

Follow the development guidelines below.

### 4. Test Your Changes

```bash
# Run the application
python app.py

# Test the modules
python -m pytest tests/  # If tests exist
```

### 5. Commit Your Changes

Use conventional commit messages:
```
feat: add new bias detection algorithm
fix: resolve memory leak in data loader
docs: update installation guide
test: add unit tests for preprocessing
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## 📝 Pull Request Process

### Before Submitting

- [ ] Code follows the project's coding standards
- [ ] Self-review of your code has been performed
- [ ] Code has been commented, particularly in hard-to-understand areas
- [ ] No sensitive information (API keys, passwords) is included
- [ ] Corresponding changes to documentation have been made
- [ ] Changes generate no new warnings or errors
- [ ] Security scan passes

### PR Requirements

1. **Title**: Use conventional commit format
2. **Description**: Clearly describe what changes were made and why
3. **Link Issues**: Reference any related issues
4. **Screenshots**: If applicable, include screenshots of changes
5. **Testing**: Describe how you tested your changes

### Review Process

1. **Automated Checks**: PR must pass all automated validations
2. **Code Review**: Repository owner will review the code
3. **Security Review**: Security implications will be assessed
4. **Approval**: Only the repository owner can approve and merge PRs

## 🛠️ Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions small and focused
- Use type hints where appropriate

### File Organization

```
FACT-TRACK/
├── app.py                 # Main application
├── config.py             # Configuration
├── modules/              # Core modules
│   ├── bert_bias_model.py
│   ├── bert_category_model.py
│   └── ...
├── scripts/              # Utility scripts
├── docs/                 # Documentation
└── tests/                # Test files
```

### Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions
- Update API documentation if applicable
- Include examples for new features

## 🔐 Security Guidelines

### What NOT to Include

- API keys or secrets
- Personal information
- Hardcoded credentials
- Sensitive configuration data
- Database connection strings

### Security Best Practices

- Use environment variables for sensitive data
- Validate all user inputs
- Implement proper error handling
- Keep dependencies updated
- Follow OWASP guidelines

### Security Checklist

- [ ] No secrets in code
- [ ] Input validation implemented
- [ ] Error handling doesn't leak information
- [ ] Dependencies are up to date
- [ ] No SQL injection vulnerabilities
- [ ] Proper authentication/authorization

## 🐛 Reporting Issues

### Bug Reports

When reporting bugs, include:

1. **Clear Title**: Brief description of the issue
2. **Description**: Detailed explanation of the problem
3. **Steps to Reproduce**: Exact steps to reproduce the issue
4. **Expected Behavior**: What should happen
5. **Actual Behavior**: What actually happens
6. **Environment**: OS, Python version, dependencies
7. **Screenshots**: If applicable

### Feature Requests

For feature requests:

1. **Use Case**: Describe the problem you're trying to solve
2. **Proposed Solution**: Your suggested approach
3. **Alternatives**: Other solutions you've considered
4. **Additional Context**: Any other relevant information

### Security Issues

**DO NOT** create public issues for security vulnerabilities. Instead:

1. Email: khizer.farhaan7@gmail.com
2. Subject: "Security Issue in FACT-TRACK"
3. Include detailed description and steps to reproduce

## 📚 Resources

- [Python Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [Security Best Practices](https://owasp.org/www-project-top-ten/)

## 🤔 Questions?

If you have questions:

1. Check existing issues and discussions
2. Create a new issue with the "question" label
3. Email: khizer.farhaan7@gmail.com

## 📄 License

By contributing to FACT-TRACK, you agree that your contributions will be licensed under the same license as the project.

---

**Thank you for contributing to FACT-TRACK!** 🎉
