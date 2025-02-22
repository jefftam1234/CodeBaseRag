from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="CodeBaseRag",
    version="0.1.0",
    description="A local RAG system for multi-language codebases using Qdrant and local LLMs",
    author="Jeff Tam",
    author_email="jefftam1234@gmail.com",
    url="https://github.com/jefftam1234/CodeBaseRAG",
    packages=find_packages(include=["src", "src.*", "user_interface", "user_interface.*"]),
    install_requires=required,
    entry_points={
        "console_scripts": [
            "codebaserag-menu=main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
