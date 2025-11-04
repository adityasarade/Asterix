# Publishing Guide

This document describes the process for building and publishing the Asterix Agent library to PyPI.

## Prerequisites

### 1. Install Build Tools
```bash
pip install --upgrade pip
pip install build twine
```

### 2. PyPI Account Setup

- Create account at https://pypi.org/account/register/
- Verify your email
- Set up 2FA (recommended)
- Create an API token at https://pypi.org/manage/account/token/

### 3. Configure PyPI Credentials

Create/edit `~/.pypirc`:
```ini
[distutils]
index-servers =
    pypi

[pypi]
username = __token__
password = pypi-YOUR-API-TOKEN-HERE
```

**Security Note:** Never commit `.pypirc` to git. It's already in `.gitignore`.

---

## Pre-Release Checklist

Before building and publishing, verify:

- [ ] All tests passing: `pytest`
- [ ] Version updated in `pyproject.toml`
- [ ] CHANGELOG.md updated with release notes
- [ ] README.md is current
- [ ] All example scripts tested and working
- [ ] No debug code or print statements in production code
- [ ] Dependencies versions are correct
- [ ] License file is present
- [ ] Git tag created for the release

---

## Building the Package

### 1. Clean Previous Builds
```bash
# Remove old build artifacts
rm -rf dist/ build/ *.egg-info
```

### 2. Build Distribution Packages
```bash
# Build source distribution and wheel
python -m build
```

This creates:
- `dist/asterix-agent-X.X.X.tar.gz` (source distribution)
- `dist/asterix_agent-X.X.X-py3-none-any.whl` (wheel)

### 3. Verify Build Contents
```bash
# Check what's included in the package
tar -tzf dist/asterix-agent-*.tar.gz

# Or for wheel:
unzip -l dist/asterix_agent-*.whl
```

Verify that:
- ✅ All Python files are included
- ✅ README.md, CHANGELOG.md, LICENSE are present
- ✅ Examples directory is included
- ✅ No unnecessary files (.pyc, __pycache__, .env)

---

## Publishing to PyPI

### Test PyPI First (Recommended)

Test on PyPI's test server before publishing to production:
```bash
# Upload to Test PyPI
python -m twine upload --repository testpypi dist/*

# Install from Test PyPI to verify
pip install --index-url https://test.pypi.org/simple/ asterix-agent
```

### Publish to Production PyPI

Once verified on Test PyPI:
```bash
# Upload to PyPI
python -m twine upload dist/*
```

You'll be prompted for credentials (use API token).

---

## Post-Release Steps

### 1. Create Git Tag
```bash
# Tag the release
git tag -a v0.1.0 -m "Release version 0.1.0"

# Push tag to GitHub
git push origin v0.1.0
```

### 2. Create GitHub Release

1. Go to https://github.com/adityasarade/Asterix/releases
2. Click "Create a new release"
3. Select the tag you created
4. Title: "v0.1.0 - Initial Stable Release"
5. Copy release notes from CHANGELOG.md
6. Publish release

### 3. Verify Installation
```bash
# Wait a few minutes for PyPI to update, then test install
pip install --upgrade asterix-agent

# Verify version
python -c "import asterix; print(asterix.__version__)"
```

### 4. Update Documentation

- [ ] Update README.md if needed
- [ ] Update installation instructions
- [ ] Announce release (Twitter, Discord, etc.)

---

## Version Numbering

Follow Semantic Versioning (https://semver.org/):

- **Major (X.0.0)**: Breaking changes
- **Minor (0.X.0)**: New features, backwards compatible
- **Patch (0.0.X)**: Bug fixes, backwards compatible

Examples:
- `0.1.0` → `0.1.1`: Bug fix release
- `0.1.0` → `0.2.0`: New features added
- `0.9.0` → `1.0.0`: First stable release with potential breaking changes

---

## Troubleshooting

### Build Fails
```bash
# Check for syntax errors
python -m py_compile asterix/**/*.py

# Verify pyproject.toml syntax
python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"
```

### Upload Fails

**Error: "File already exists"**
- You cannot re-upload the same version
- Increment version in `pyproject.toml` and rebuild

**Error: "Invalid credentials"**
- Check `~/.pypirc` configuration
- Verify API token is correct
- Ensure username is `__token__` (not your PyPI username)

### Import Fails After Install
```bash
# Check package contents
pip show -f asterix-agent

# Verify __init__.py exports
python -c "import asterix; print(dir(asterix))"
```

---

## Quick Reference

### Complete Publishing Workflow
```bash
# 1. Update version in pyproject.toml
# 2. Update CHANGELOG.md
# 3. Commit changes
git add .
git commit -m "Release v0.1.0"

# 4. Clean and build
rm -rf dist/ build/ *.egg-info
python -m build

# 5. Check contents
tar -tzf dist/asterix-agent-*.tar.gz

# 6. Test upload (optional)
python -m twine upload --repository testpypi dist/*

# 7. Upload to PyPI
python -m twine upload dist/*

# 8. Tag and push
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin main
git push origin v0.1.0

# 9. Create GitHub release
# Go to GitHub and create release from tag

# 10. Verify
pip install --upgrade asterix-agent
python -c "import asterix; print(asterix.__version__)"
```

---

## Security Notes

- ✅ Use API tokens instead of passwords
- ✅ Enable 2FA on your PyPI account
- ✅ Never commit `.pypirc` or API tokens to git
- ✅ Use Test PyPI for testing before production
- ✅ Review package contents before uploading

---

## Resources

- **PyPI**: https://pypi.org/project/asterix-agent/
- **Test PyPI**: https://test.pypi.org/
- **Python Packaging Guide**: https://packaging.python.org/
- **Twine Documentation**: https://twine.readthedocs.io/
- **Semantic Versioning**: https://semver.org/

---

**Last Updated:** 2025-11-04