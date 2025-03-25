"""
Setup script for the Note Organizer package.
"""

from setuptools import setup, find_packages

# Read requirements
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

# Read the README for long description
with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="note_organizer",
    version="0.1.0",
    description="A tool for organizing and tagging markdown notes with semantic search and auto-tagging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Note Organizer Team",
    author_email="info@noteorganizer.io",
    url="https://github.com/noteorganizer/note-organizer",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Text Processing :: Markup :: Markdown",
        "Topic :: Office/Business :: Personal Information Managers",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "note-organizer=note_organizer.__main__:main",
        ],
    },
) 