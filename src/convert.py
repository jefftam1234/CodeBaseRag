#!/usr/bin/env python3
import os
import argparse
from user_interface.config import config

def convert_files_to_txt(src_dir, dst_dir, extensions=(".py", ".cpp", ".c", ".h", ".hpp", ".java", ".md", ".txt")):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(extensions):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, src_dir)
                new_root = os.path.join(dst_dir, os.path.dirname(rel_path))
                os.makedirs(new_root, exist_ok=True)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            data = f.read()
                    except UnicodeDecodeError:
                        print("Failed to decode: " + file_path)
                        continue
                new_file_path = os.path.join(new_root, file + '.txt')
                with open(new_file_path, 'w', encoding='utf-8') as f:
                    f.write(data)
    print("Conversion complete.")

def main():
    parser = argparse.ArgumentParser(description="Convert source code files to plain text.")
    parser.add_argument("--src", type=str, default=config.DEFAULT_CODEBASE_PATH,
                        help="Source directory containing code files (default from config)")
    parser.add_argument("--dst", type=str, default=config.DEFAULT_CONVERTED_PATH,
                        help="Destination directory for converted text files (default from config)")
    args = parser.parse_args()
    convert_files_to_txt(args.src, args.dst)

if __name__ == "__main__":
    main()
