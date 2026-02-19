"""Setup configuration for media-mix-modeling package."""

from setuptools import setup, find_packages

setup(
    name="media-mix-modeling",
    version="1.0.0",
    description="Databricks Solution Accelerator for Media Mix Modeling using PyMC-Marketing",
    author="Databricks",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.10",
    install_requires=[
        "graphviz==0.20.3",
        "mlflow==3.6.0",
        "pymc-marketing==0.17.1",
    ],
    extras_require={
        "dev": [
            "pytest",
            "jupyter",
            "ipywidgets",
        ]
    },
)
