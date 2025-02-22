#!/usr/bin/env python3
import argparse
import pickle
from user_interface.config import config
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(input_file, output_file, chunk_size, chunk_overlap):
    with open(input_file, "rb") as f:
        documents = pickle.load(f)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc_chunks = text_splitter.split_documents(documents)
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
    parser.add_argument("--chunk_size", type=int, default=1500, help="Chunk size (default: 1500)")
    parser.add_argument("--chunk_overlap", type=int, default=150, help="Chunk overlap (default: 150)")
    args = parser.parse_args()
    split_documents(args.input, args.output, args.chunk_size, args.chunk_overlap)

if __name__ == "__main__":
    main()
