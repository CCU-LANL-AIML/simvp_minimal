from setuptools import setup, find_packages
import platform

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Adjust PyTorch and NumPy versions for Intel-based macOS
if platform.system() == "Darwin" and platform.machine() == "x86_64":
    requirements = [
        r if not r.startswith("torch==") else "torch<=2.2.2" for r in requirements
    ]
    requirements = [
        r if not r.startswith("numpy==") else "numpy<2.0" for r in requirements
    ]

setup(
    name="simvp_minimal",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    description="Minimal implementation of SimVP for spatiotemporal prediction",
    author="",
    author_email="",
)