# CodeBaseRag: A Local RAG System for Multi-Language Codebases

Welcome to **CodeBaseRag**, a local Recency-Based Aggregation (RAG) system designed for working with codebases in multiple programming languages, such as Python, R, and JavaScript. This system integrates with Qdrant for vector search and supports various document formats like PDF, HTML, and Markdown.

## Features

- **Support Multiple Languages**: Work with codebases written in Python, R, or JavaScript.
- **Document Formats**: Process and query documents in formats like PDF, HTML, Markdown, JSON, and more.
- **Qdrant Integration**: Utilize Qdrant for efficient vector search and similarity-based retrieval.
- **Local LLMs**: Leverage local pre-trained language models using LangChain and Transformers.
- **Interactive Interface**: Access an interactive command-line interface (CLI) or Gradio-based graphical user interface (GUI).

## Installation

To install and set up CodeBaseRag, follow these steps:

### Prerequisites
1. Ensure Python 3.8 or higher is installed.
2. Install required dependencies:
   ```bash
   pip install .
   ```

### Steps to Set Up

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/CodeBaseRag.git
   cd CodeBaseRag
   ```

you can install locally with 
```bash
pip install .
```

or you can create a standalone executable with pyinstaller
```bash
pip install pyinstaller
pyinstaller --onefile main.py
```


2. **Create Configuration Files**:
   - Copy `config.template.ini` to `config.ini`:
     ```bash
     cp config.template.ini config.ini
     ```
     and then edit the config.ini to your own settings, and put it in the active directory

3. **Configure Your Settings**:
   - **Language-Specific Configurations**: Update parameters based on your preferred programming language (Python, R, JavaScript) in the respective language-specific configuration files.
   - **Qdrant Configuration**: Configure Qdrant settings such as host, port, collection names, and vector dimensions in `config qdrant.ini`.

4. **Make sure you have Docker installed**

5. **Set Up Interactive Interface**:
   - To access the CLI interface, run:
     ```bash
     codebaserag
     ```
   - Replace the default prompt with your custom prompt by editing `src/cli.py` and updating the `query prompt string` variable.

## Using the Interactive CLI or GUI

### CLI Interface

- Launch the CLI by running `codebaserag`.
- Enter your queries one by one, pressing '/exit' to return to the main menu.
- Each query will trigger a request to Qdrant for relevant documents and an interaction with the LLM model specified in your configuration.



### maintain requirement
```
pip install pipreqs
pipreqs . --force
```