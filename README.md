Below is your README.md with improved formatting and consistent indentation:

---

# CodeBaseRag: A Local RAG System for Multi-Language Codebases

**CodeBaseRag** is a local Recency-Based Aggregation (RAG) system designed to work with codebases written in Python, R, and JavaScript. It integrates with Qdrant for vector search and supports a variety of document formats, including PDF, HTML, Markdown, and JSON.

## Features

- **Multi-Language Support**: Works with Python, R, and JavaScript codebases.
- **Document Processing**: Supports formats such as PDF, HTML, Markdown, JSON, and more.
- **Qdrant Integration**: Utilizes Qdrant for efficient vector search and similarity-based retrieval.
- **Local LLMs**: Leverages pre-trained language models through LangChain and Transformers.
- **User Interfaces**: Choose between an interactive CLI and a Gradio-based GUI.

## Installation

### Prerequisites

- Python 3.8 or higher
- Docker installed on your system

### Setup Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/jefftam1234/CodeBaseRag.git
   cd CodeBaseRag
   ```

2. **Install the Package**
   - To install locally:
     ```bash
     pip install .
     ```

3. **Create and Edit Configuration Files**
   - Copy the template configuration file:
     ```bash
     cp config.template.ini config.ini
     ```
   - Edit `config.ini` with your settings and place it in the active directory.

4. **Configure Settings**
   - **Qdrant Settings**: In `config qdrant.ini`, specify your Qdrant host, port, collection names, and vector dimensions.
   - **Local Paths**: Adjust paths such as `DEFAULT_CODEBASE_PATH` and `DEFAULT_QDRANT_STORAGE_FOLDER` to match your environment.

5. **Run the Main Menu**
   ```bash
   codebaserag-menu
   ```

   The main menu provides five options:
   1. Convert files to text and perform chunking/splitting.
   2. Ensure Docker is installed to run Qdrant (the vectorized database). Press “l” to launch and “k” to kill.
   3. Push the chunked files to Qdrant.
   4. In the GUI, choose between a command-line and a graphical interface. The GUI lets you select your installed LLM and the collection (the pushed code base).
   5. Load any available configuration files; launching the main code again will use the selected configuration.

## Running the Application

### CLI Interface

- Launch the CLI by running:
  ```bash
  codebaserag
  ```
- Enter your queries one at a time. Use `/exit` to return to the main menu.
- Each query triggers a vector search in Qdrant and interacts with the configured local LLM.

### GUI Interface (Gradio)

- Launch the Gradio-based GUI according to your configuration.

## Configuration Template

Below is an excerpt from the `config.template.ini` to help you get started:

```ini
[DEFAULT]
# Qdrant settings
DEFAULT_QDRANT_HOST = localhost
DEFAULT_QDRANT_PORT = 6333

# Collection name for Qdrant (required)
DEFAULT_COLLECTION_NAME = your_code_base

# Paths (required)
DEFAULT_CODEBASE_PATH = /home/your_code_base
# These paths are computed at runtime:
# DEFAULT_CONVERTED_PATH = <computed at runtime>
# DEFAULT_DOCS_PICKLE = <computed at runtime>
# DEFAULT_CHUNKS_PICKLE = <computed at runtime>

# Qdrant storage folder (required)
DEFAULT_QDRANT_STORAGE_FOLDER = /home/qdrant_storage
# DEFAULT_CONTAINER_ID_FILE is computed at runtime.

# LLM model: Set your default LLM model here.
DEFAULT_LLM_MODEL = your_llm:latest
# Alternative models:
# DEFAULT_LLM_MODEL = deepseek-r1:latest
# DEFAULT_LLM_MODEL = codellama:7b

# Gradio settings
DEFAULT_GRADIO_SHARE = False
DEFAULT_GRADIO_SERVER_NAME = 0.0.0.0
DEFAULT_GRADIO_SERVER_PORT = 7860
```

## Maintain Requirements

- Install and run pipreqs to generate or update your requirements file:
  ```bash
  pip install pipreqs
  pipreqs . --force
  ```
