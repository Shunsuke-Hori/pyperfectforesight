"""Setup script for dynare_python package"""

from setuptools import setup, find_packages
from pathlib import Path

# Read version
version = {}
with open("dynare_python/__version__.py") as f:
    exec(f.read(), version)

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")
else:
    long_description = "A minimal Dynare-style perfect foresight solver in Python"

setup(
    name="dynare-python",
    version=version["__version__"],
    description="A minimal Dynare-style perfect foresight solver in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/DynareByPython",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "sympy>=1.9.0",
    ],
    extras_require={
        "dev": [
            "matplotlib>=3.3.0",
            "pytest>=6.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="economics macroeconomics dynare perfect-foresight",
)
