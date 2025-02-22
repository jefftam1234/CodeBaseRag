#!/usr/bin/env python3
import argparse
from user_interface.config import config
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from src.embeddings import get_embeddings
from src.llm import OllamaLLM


def query(query: str, host: str, port: int, collection_name: str, model: str, suppress_output = False) -> str:
    # Instantiate embeddings (with CUDA if available)
    embeddings = get_embeddings(suppress_output=suppress_output)

    # Create a Qdrant client pointing to your running server
    client = QdrantClient(host=host, port=port)

    # Create a QdrantVectorStore (no external pickled index)
    qdrant_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )

    # Create a retriever from the vector store
    retriever = qdrant_store.as_retriever(search_kwargs={"k": 3})

    # Get the custom LLM singleton instance using the model from config or command line.
    llm = OllamaLLM.get_instance(model, verbose=not suppress_output)

    # Build a RetrievalQA chain using the retriever and LLM.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

    # Process the query using the new invoke method
    qa_response = qa_chain.invoke({"query": query})
    return qa_response["result"]


def main():
    parser = argparse.ArgumentParser(
        description="Query the codebase directly from Qdrant using the RAG system."
    )
    # Only the query is required; the rest default from config.
    parser.add_argument("query", help="The query to ask about the codebase.")
    parser.add_argument("--host", default=config.DEFAULT_QDRANT_HOST,
                        help="Qdrant server host (default from config).")
    parser.add_argument("--port", type=int, default=config.DEFAULT_QDRANT_PORT,
                        help="Qdrant server port (default from config).")
    parser.add_argument("--collection", default=config.DEFAULT_COLLECTION_NAME,
                        help="Collection name (default from config).")
    parser.add_argument("--model", default=config.DEFAULT_LLM_MODEL,
                        help="LLM model to use (default from config).")
    args = parser.parse_args()

    answer = query(args.query, args.host, args.port, args.collection, args.model)
    print("Answer:", answer)


if __name__ == "__main__":
    main()
