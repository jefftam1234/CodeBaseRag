#!/usr/bin/env python3
from langchain_huggingface import HuggingFaceEmbeddings
from user_interface.config import config

def get_embeddings(suppress_output: bool = False):
    # Use the default device from config (either 'cuda' or 'cpu')
    device = config.DEFAULT_DEVICE
    if not suppress_output:
        print(f"Using device: {device}")
    embeddings = HuggingFaceEmbeddings(
        #model_name="BAAI/bge-base-en-v1.5",
        model_name="all-mpnet-base-v2",
        #TODO: use https://huggingface.co/microsoft/unixcoder-base
        model_kwargs={"device": device}
    )
    return embeddings

if __name__ == "__main__":
    emb = get_embeddings()
    test_vec = emb.embed_query("Sample query for testing embeddings.")
    print("Sample embedding vector:", test_vec)
