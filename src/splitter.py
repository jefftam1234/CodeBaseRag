#!/usr/bin/env python3
import argparse
import pickle
import re, os
from user_interface.config import config
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.text_splitter import MarkdownTextSplitter


class MatlabSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Define a regex pattern that captures MATLAB function declarations, class definitions, subroutines, or script sections.
        self.boundary_pattern = re.compile(r"^\s*(function|classdef|sub|%%)\b", re.IGNORECASE)

    def split_documents(self, docs: list) -> list:
        chunks = []
        for doc in docs:
            lines = doc.page_content.splitlines()
            current_chunk = ""
            for line in lines:
                if self.boundary_pattern.match(line) and current_chunk.strip():
                    if len(current_chunk) >= self.chunk_size:
                        chunks.append(current_chunk)
                        # Start new chunk: include overlap.
                        current_chunk = current_chunk[-self.chunk_overlap:] + "\n" + line
                        continue
                    else:
                        chunks.append(current_chunk)
                        current_chunk = line
                else:
                    current_chunk += "\n" + line
            if current_chunk.strip():
                chunks.append(current_chunk)
        return chunks


class JuliaSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Define a regex pattern for Julia boundaries: functions, struct, module, abstract type, etc.
        self.boundary_pattern = re.compile(r"^\s*(function|struct|module|abstract type)\b", re.IGNORECASE)

    def split_documents(self, docs: list) -> list:
        chunks = []
        for doc in docs:
            lines = doc.page_content.splitlines()
            current_chunk = ""
            for line in lines:
                if self.boundary_pattern.match(line) and current_chunk.strip():
                    if len(current_chunk) >= self.chunk_size:
                        chunks.append(current_chunk)
                        current_chunk = current_chunk[-self.chunk_overlap:] + "\n" + line
                        continue
                    else:
                        chunks.append(current_chunk)
                        current_chunk = line
                else:
                    current_chunk += "\n" + line
            if current_chunk.strip():
                chunks.append(current_chunk)
        return chunks


# Mapping from file extension (without dot) to language key.
# For any file extension that is not recognized, we'll use "markdown".
EXTENSION_TO_LANGUAGE = {
    "cpp": "cpp",
    "c": "cpp",
    "hpp": "cpp",
    "h": "cpp",
    "java": "java",
    "cs": "csharp",
    "csharp": "csharp",
    "jl": "julia",
    "julia": "julia",
    "m": "matlab",
    "mat": "matlab",
    "md": "markdown",
    "txt": "default",  # Plain text files (if you want them generic)
}


def get_language_splitter(language, chunk_size, chunk_overlap):
    if language == "python":
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif language == "markdown":
        return MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif language == "cpp":
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.CPP, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif language == "java":
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.JAVA, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif language == "julia":
        return JuliaSplitter(chunk_size, chunk_overlap)
    elif language == "matlab":
        return MatlabSplitter(chunk_size, chunk_overlap)
    elif language == "csharp":
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.CSHARP, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        # If for some reason, it still falls through, use Markdown as a fallback.
        return MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def split_documents(input_file, output_file, chunk_size, chunk_overlap, language_splitting):
    with open(input_file, "rb") as f:
        documents = pickle.load(f)

    if language_splitting:
        # Build a dictionary of splitters for each supported language in your config.
        # Here, config.CODEBASE_LANGUAGES is expected to be a list of language keys,
        # e.g. ["cpp", "java", "python", "matlab", "csharp", "julia", "markdown"]
        splitters = {lang: get_language_splitter(lang, chunk_size, chunk_overlap) for lang in config.CODEBASE_LANGUAGES}
        # Add a default fallback splitter.
        splitters["default"] = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        generic_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splitters = {"default": generic_splitter}

    doc_chunks = []
    for doc in documents:
        # Assume doc.metadata["source"] is like "/path/to/file.ext.txt"
        source = doc.metadata["source"]
        # Remove the appended ".txt"
        base, ext1 = os.path.splitext(source)
        if ext1.lower() == ".txt":
            base, ext2 = os.path.splitext(base)
            ext = ext2.lstrip(".").lower()
        else:
            ext = ext1.lstrip(".").lower()

        # Map file extension to language key
        lang_key = EXTENSION_TO_LANGUAGE.get(ext, "default")
        splitter = splitters.get(lang_key, splitters["default"])
        doc_chunks.extend(splitter.split_documents([doc]))

    for doc in doc_chunks:
        doc.metadata["source"] = doc.metadata["source"].replace(".txt", "")

    with open(output_file, "wb") as f:
        pickle.dump(doc_chunks, f)
    print(f"Split into {len(doc_chunks)} chunks and saved to {output_file}.")


def main():
    parser = argparse.ArgumentParser(description="Split documents into chunks.")
    parser.add_argument("--input", type=str, default=config.DEFAULT_DOCS_PICKLE,
                        help="Input pickle file (default from config)")
    parser.add_argument("--output", type=str, default=config.DEFAULT_CHUNKS_PICKLE,
                        help="Output pickle file (default from config)")
    parser.add_argument("--chunk_size", type=int, default=config.CHUNK_SIZE,
                        help="Chunk size (default from config)")
    parser.add_argument("--chunk_overlap", type=int, default=config.CHUNK_OVERLAP,
                        help="Chunk overlap (default from config)")
    parser.add_argument("--language_splitting", action="store_true", default=config.LANGUAGE_AWARE_SPLITTING,
                        help="Enable language-aware splitting (default from config)")
    args = parser.parse_args()

    split_documents(args.input, args.output, args.chunk_size, args.chunk_overlap, args.language_splitting)


if __name__ == "__main__":
    main()
