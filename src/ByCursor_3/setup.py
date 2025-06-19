#!/usr/bin/env python3
"""
CISF (Cross-Industry Synergy Finder) セットアップ
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cisf",
    version="1.0.0",
    author="CISF Development Team",
    author_email="developer@cisf.ai",
    description="Cross-Industry Synergy Finder - AI-powered brand collaboration discovery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/cisf",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "gpu": [
            "torch-scatter>=2.1.0",
            "torch-sparse>=0.6.0",
            "torch-cluster>=1.6.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "cisf=scripts.run_full_cisf_with_advanced_methods:main",
            "cisf-dashboard=src.ui.interactive_dashboard:main",
        ],
    },
    include_package_data=True,
    package_data={
        "cisf": ["config/*.yaml", "data/processed/*.parquet"],
    },
)
