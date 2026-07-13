from setuptools import setup, find_packages

setup(
    name="shallow_water",
    version="0.1.0",
    author="Spencer A. Hill",
    author_email="shill1@ccny.cuny.edu",
    description="Shallow water model, originally from Sobel and Schneider 2009",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zpcllyj/SobelSchneiderModel",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "xarray",
        "pytest",
    ],
    extras_require={
        "numba": ["numba"],
    },
    entry_points={
        "console_scripts": [
            "run-sw-model=ss09.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
