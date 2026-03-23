from pathlib import Path

from setuptools import setup, find_packages

def read_requirements():
    """Read requirements file and handle different encodings."""
    encodings = ['utf-8', 'utf-16-le', 'utf-16']
    
    for encoding in encodings:
        try:
            with open("requirements.txt", "r", encoding=encoding) as f:
                # Strip BOM and whitespace from each line
                return [line.strip().replace('\ufeff', '') 
                       for line in f 
                       if line.strip() and not line.startswith('#')]
        except UnicodeDecodeError:
            continue
    
    print("Error: Could not read requirements.txt with any supported encoding")
    return []

requirements = read_requirements()
readme_path = Path("README.md")
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="lm_against_hate",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    author="YenYu Chang",
    author_email="yenyu.chang@hotmail.com",
    description="Training and evaluation utilities for target-aware counterspeech generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
)
