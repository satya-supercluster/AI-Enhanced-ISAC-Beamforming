from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-enhanced-isac-beamforming",
    version="1.0.0",
    author="Satyam Gupta",
    author_email="your.email@university.edu",
    description="AI-Enhanced Beamforming for Energy-Efficient ISAC Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/AI-Enhanced-ISAC-Beamforming",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: System :: Networking",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=4.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "gpu": [
            "cupy>=10.0.0",
        ],
        "hardware": [
            "pyserial>=3.5",
            "pyvisa>=1.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "isac-train=src.training.train_drl:main",
            "isac-eval=src.training.evaluate:main",
            "isac-demo=examples.quick_start_demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.csv", "*.txt"],
    },
    keywords="ISAC, beamforming, deep reinforcement learning, energy efficiency, V2X, wireless communication",
    project_urls={
        "Bug Reports": "https://github.com/your-username/AI-Enhanced-ISAC-Beamforming/issues",
        "Source": "https://github.com/your-username/AI-Enhanced-ISAC-Beamforming",
        "Documentation": "https://ai-enhanced-isac-beamforming.readthedocs.io/",
    },
)