from setuptools import setup, find_packages

setup(
    name="sameerkhan",  # Package name
    version="0.1.0",    # Version
    author="Sameer Khan",
    author_email="your.email@example.com",
    description="A sample Python package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),  # Automatically find sub-packages
    install_requires=[         # Dependencies (if any)
        "numpy",
        "pandas"
    ],
    classifiers=[              # Optional metadata
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",    # Minimum Python version
)
