#!/usr/bin/env python3
import argparse
import pickle
from user_interface.config import config
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from src.embeddings import get_embeddings

def push_documents_to_qdrant(
    pickle_file: str,
    collection_name: str,
    host: str = None,
    port: int = None
):
    if host is None:
        host = config.DEFAULT_QDRANT_HOST
    if port is None:
        port = config.DEFAULT_QDRANT_PORT

    if not collection_name:
        raise ValueError("You must specify a collection_name for your codebase.")

    # Load the pickled document chunks
    with open(pickle_file, "rb") as f:
        doc_chunks = pickle.load(f)
    print(f"Loaded {len(doc_chunks)} document chunks from {pickle_file}.")

    # Instantiate the embedding model (using the function from embeddings.py)
    embeddings = get_embeddings()

    # Create a Qdrant client connecting to your Qdrant server
    client = QdrantClient(host=host, port=port)

    # Check if the collection exists; if not, create it
    try:
        client.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' exists.")
    except Exception:
        print(f"Collection '{collection_name}' does not exist, creating it...")
        # Create a dummy vector to determine the dimension of embeddings.
        dummy_vector = embeddings.embed_query("dummy")
        vector_dim = len(dummy_vector)
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"size": vector_dim, "distance": "Cosine"}
        )
        print(f"Collection '{collection_name}' created successfully.")

    # Initialize the QdrantVectorStore
    qdrant_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )

    # Prepare texts and metadata from the document chunks
    texts = [doc.page_content for doc in doc_chunks]
    metadatas = [doc.metadata for doc in doc_chunks]

    # Add the texts (and metadata) to the Qdrant store
    qdrant_store.add_texts(texts=texts, metadatas=metadatas)
    print(f"Pushed {len(doc_chunks)} document chunks to collection '{collection_name}' on {host}:{port}.")

def main():
    parser = argparse.ArgumentParser(
        description="Push document chunks (from a pickle file) to a Qdrant collection."
    )
    # Use nargs="?" to make the positional argument optional.
    parser.add_argument("pickle_file", nargs="?", default=config.DEFAULT_CHUNKS_PICKLE,
                        help="Path to the pickle file (default from config).")
    parser.add_argument("--collection_name", default=config.DEFAULT_COLLECTION_NAME,
                        help="Name of the collection (default from config).")
    parser.add_argument("--host", default=None, help="Qdrant server host (default from config).")
    parser.add_argument("--port", type=int, default=None, help="Qdrant server port (default from config).")

    args = parser.parse_args()
    push_documents_to_qdrant(
        args.pickle_file,
        collection_name=args.collection_name,
        host=args.host,
        port=args.port
    )

if __name__ == "__main__":
    main()
