from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="mosq",
    version="1.0.0",
    author="Riandanis",
    description=(
        "MoSQ: Modality-Slot Q-Former for Gastric Cancer Drug Response Prediction"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Riandanis/MoSQ",
    packages=find_packages(exclude=["tests*", "scripts*", "docs*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    entry_points={
        "console_scripts": [
            "mosq=scripts.main:main",
        ],
    },
)
