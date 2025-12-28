#!/bin/bash

# Script to remove all Python cache files
# Usage: ./clean_pycache.sh

echo "Cleaning Python cache files..."

# Remove all __pycache__ directories
echo "Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Remove all .pyc files
echo "Removing .pyc files..."
find . -name "*.pyc" -delete 2>/dev/null

# Remove all .pyo files
echo "Removing .pyo files..."
find . -name "*.pyo" -delete 2>/dev/null

# Remove all .pyd files (Windows)
echo "Removing .pyd files..."
find . -name "*.pyd" -delete 2>/dev/null

# Remove all .py[cod] files
echo "Removing .py[cod] files..."
find . -name "*.py[cod]" -delete 2>/dev/null

# Remove all .so files (compiled extensions)
echo "Removing .so files..."
find . -name "*.so" -delete 2>/dev/null

# Remove all .egg-info directories
echo "Removing .egg-info directories..."
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null

# Remove all .pytest_cache directories
echo "Removing .pytest_cache directories..."
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null

# Remove all .coverage files
echo "Removing .coverage files..."
find . -name ".coverage" -delete 2>/dev/null

# Remove all .coverage.* files
echo "Removing .coverage.* files..."
find . -name ".coverage.*" -delete 2>/dev/null

# Remove all .cache directories
echo "Removing .cache directories..."
find . -type d -name ".cache" -exec rm -rf {} + 2>/dev/null

# Remove all .tox directories
echo "Removing .tox directories..."
find . -type d -name ".tox" -exec rm -rf {} + 2>/dev/null

# Remove all .nox directories
echo "Removing .nox directories..."
find . -type d -name ".nox" -exec rm -rf {} + 2>/dev/null

# Remove all .mypy_cache directories
echo "Removing .mypy_cache directories..."
find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null

# Remove all .ruff_cache directories
echo "Removing .ruff_cache directories..."
find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null

echo "Python cache cleanup complete!"
echo "Removed:"
echo "  - __pycache__ directories"
echo "  - .pyc files"
echo "  - .pyo files"
echo "  - .pyd files"
echo "  - .py[cod] files"
echo "  - .so files"
echo "  - .egg-info directories"
echo "  - .pytest_cache directories"
echo "  - .coverage files"
echo "  - .cache directories"
echo "  - .tox directories"
echo "  - .nox directories"
echo "  - .mypy_cache directories"
echo "  - .ruff_cache directories"