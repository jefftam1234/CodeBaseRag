import configparser
import logging
import os
import torch
import ast
from pydantic import BaseModel, Field, ValidationError


class AppConfig(BaseModel):
    DEFAULT_DEVICE: str = Field("cuda" if torch.cuda.is_available() else "cpu")
    # Required fields
    DEFAULT_CODEBASE_PATH: str = Field(..., min_length=1)
    DEFAULT_QDRANT_HOST: str = Field("localhost")
    DEFAULT_QDRANT_PORT: int = Field(6333)
    DEFAULT_COLLECTION_NAME: str = Field(..., min_length=1)
    DEFAULT_LLM_MODEL: str = Field(..., min_length=1)
    DEFAULT_QDRANT_STORAGE_FOLDER: str = Field(..., min_length=1)

    # Optional fields – will be computed if not provided.
    DEFAULT_CONVERTED_PATH: str = None
    DEFAULT_DOCS_PICKLE: str = None
    DEFAULT_CHUNKS_PICKLE: str = None
    DEFAULT_CONTAINER_ID_FILE: str = None
    DEFAULT_GRADIO_SHARE: bool = Field(False)
    DEFAULT_GRADIO_SERVER_NAME: str = Field("0.0.0.0")
    DEFAULT_GRADIO_SERVER_PORT: int = Field(7860)
    DEFAULT_OLLAMA_HOST: str = Field("http://localhost:11434")

    # New fields for retrieval and splitting tuning:
    LANGUAGE_AWARE_SPLITTING: bool = Field(True, description="Enable language-aware splitting")
    CODEBASE_LANGUAGES: list[str] = Field(["cpp", "java", "python", "matlab", "csharp", "julia", "markdown"],
                                      description="List of languages in the codebase")
    CHUNK_SIZE: int = Field(1500, description="Chunk size (in characters) for splitting documents")
    CHUNK_OVERLAP: int = Field(150, description="Overlap (in characters) between chunks")
    RETRIEVER_K: int = Field(3, description="Number of chunks to retrieve during query")

    def compute_optional(self):
            if self.DEFAULT_CODEBASE_PATH:
                # For example, instead of placing 'converted' as a subfolder, you might want
                # to place it next to the codebase directory with a "Converted" suffix.
                if not self.DEFAULT_CONVERTED_PATH or not self.DEFAULT_CONVERTED_PATH.strip():
                    self.DEFAULT_CONVERTED_PATH = os.path.join(
                        os.path.dirname(self.DEFAULT_CODEBASE_PATH),
                        os.path.basename(self.DEFAULT_CODEBASE_PATH) + "Converted"
                    )
                if not self.DEFAULT_DOCS_PICKLE or not self.DEFAULT_DOCS_PICKLE.strip():
                    self.DEFAULT_DOCS_PICKLE = os.path.join(self.DEFAULT_CONVERTED_PATH, "docs.pkl")
                if not self.DEFAULT_CHUNKS_PICKLE or not self.DEFAULT_CHUNKS_PICKLE.strip():
                    self.DEFAULT_CHUNKS_PICKLE = os.path.join(self.DEFAULT_CONVERTED_PATH, "chunks.pkl")
            else:
                raise ValueError("Invalid codebase path!")

            if self.DEFAULT_QDRANT_STORAGE_FOLDER:
                if not self.DEFAULT_CONTAINER_ID_FILE or not self.DEFAULT_CONTAINER_ID_FILE.strip():
                    self.DEFAULT_CONTAINER_ID_FILE = os.path.join(
                        os.path.dirname(self.DEFAULT_QDRANT_STORAGE_FOLDER),
                        "qdrant_container_id.txt"
                    )
            else:
                raise ValueError("Invalid Qdrant storage folder!")

def load_config_from_ini(ini_file: str = "config.ini") -> AppConfig:
    parser = configparser.ConfigParser()
    parser.optionxform = str  # preserve case
    if not os.path.exists(ini_file):
        raise FileNotFoundError(f"Config file '{ini_file}' not found!")
    parser.read(ini_file)
    data = dict(parser["DEFAULT"])

    if "DEFAULT_GRADIO_SHARE" in data:
        data["DEFAULT_GRADIO_SHARE"] = data["DEFAULT_GRADIO_SHARE"].lower() in ("true", "1", "yes")
    # Convert CODEBASE_LANGUAGES field.
    if "CODEBASE_LANGUAGES" in data:
        try:
            # Attempt to evaluate the string as a Python literal.
            data["CODEBASE_LANGUAGES"] = ast.literal_eval(data["CODEBASE_LANGUAGES"])
            # Ensure the value is a list.
            if not isinstance(data["CODEBASE_LANGUAGES"], list):
                raise ValueError("CODEBASE_LANGUAGES must be a list")
        except Exception as e:
            # Fallback: assume comma-separated values.
            data["CODEBASE_LANGUAGES"] = [lang.strip() for lang in data["CODEBASE_LANGUAGES"].split(",") if lang.strip()]

    try:
        config_instance = AppConfig(**data)
        config_instance.compute_optional()
        if config_instance.DEFAULT_CODEBASE_PATH.strip() in ["/home/your_code_base", ""]:
            import logging
            logging.warning("DEFAULT_CODEBASE_PATH is not set properly in the config file.")
        return config_instance
    except ValidationError as e:
        print("Configuration validation error:")
        print(e)
        raise e

# Load the configuration at import time from the default file.
config = load_config_from_ini()

def overwrite_config_ini(ini_file: str):
    """
    Reload the configuration from a specified INI file and update the global `config`.
    Also, overwrite the default config.ini file with the new settings.
    """
    global config
    config = load_config_from_ini(ini_file)
    # Write out the current configuration back to the default config.ini.
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser["DEFAULT"] = {key: str(value) for key, value in config.model_dump().items()}
    with open("config.ini", "w") as f:
        parser.write(f)
    print(f"[INFO] Config reloaded from {ini_file} and saved as default.")
    print(config.model_dump_json(indent=2))

def display_current_config():
    """
    Display the current configuration settings.
    """
    print(config.model_dump_json(indent=2))

if __name__ == "__main__":
    # For testing purposes: load and print the configuration.
    print("Loaded configuration:")
    display_current_config()
