from setuptools import setup, find_packages

setup(
    name="shallow_water",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Shallow water model, originally from Sobel and Schneider 2009",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zpcllyj/SobelSchneiderModel",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "xarray",
        "pytest",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
