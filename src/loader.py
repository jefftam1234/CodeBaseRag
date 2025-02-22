#!/usr/bin/env python3
import argparse
import pickle
from user_interface.config import config
from langchain_community.document_loaders import DirectoryLoader, TextLoader

def load_documents(src_dir, output_file):
    # Use glob to only load .txt files
    loader = DirectoryLoader(src_dir, glob="**/*.txt", show_progress=True, loader_cls=TextLoader)
    documents = loader.load()
    with open(output_file, "wb") as f:
        pickle.dump(documents, f)
    print(f"Loaded {len(documents)} documents and saved to {output_file}.")

def main():
    parser = argparse.ArgumentParser(
        description="Load documents from a directory and save to a pickle file."
    )
    parser.add_argument("--src", type=str, default=config.DEFAULT_CONVERTED_PATH,
                        help="Source directory of converted text files (default from config)")
    parser.add_argument("--dst", type=str, default=config.DEFAULT_DOCS_PICKLE,
                        help="Output pickle file (default from config)")
    args = parser.parse_args()
    load_documents(args.src, args.dst)

if __name__ == "__main__":
    main()
