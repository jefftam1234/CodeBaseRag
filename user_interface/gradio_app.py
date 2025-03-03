#!/usr/bin/env python3
import gradio as gr
import subprocess
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from src.embeddings import get_embeddings
from src.llm import OllamaLLM
from user_interface.config import config

def list_installed_models() -> list:
    """
    Returns a list of installed LLM model names by calling 'ollama list'.
    The first line (header) is discarded.
    """
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error listing installed LLM models:", result.stderr)
        return []
    lines = result.stdout.splitlines()
    # Skip header and extract the first column from each remaining line.
    models = [line.split()[0] for line in lines[1:] if line.strip()]
    return models

def list_qdrant_collections(host: str = config.DEFAULT_QDRANT_HOST, port: int = config.DEFAULT_QDRANT_PORT) -> list:
    try:
        client = QdrantClient(host=host, port=port)
        response = client.get_collections()
        # Try accessing the collections directly
        if hasattr(response, "result"):
            collections = [col.name for col in response.result.collections]
        else:
            collections = [col.name for col in response.collections]
        # Ensure the default collection is included.
        if config.DEFAULT_COLLECTION_NAME not in collections:
            collections.insert(0, config.DEFAULT_COLLECTION_NAME)
        return collections
    except Exception as e:
        print("Error retrieving collections:", e)
        return [config.DEFAULT_COLLECTION_NAME]

def load_documents(folder_path: str,
                   host: str = config.DEFAULT_QDRANT_HOST,
                   port: int = config.DEFAULT_QDRANT_PORT,
                   collection: str = config.DEFAULT_COLLECTION_NAME) -> str:
    """
    Dummy function to mimic document loading and pushing to Qdrant.
    In practice, this should run your conversion, loading, and splitting scripts.
    """
    # Here, you would typically process folder_path, push documents into Qdrant, etc.
    # For demonstration, we just return a confirmation message.
    return f"Documents from '{folder_path}' have been loaded and pushed to collection '{collection}' at {host}:{port}."

def answer_query(query: str,
                 host: str = config.DEFAULT_QDRANT_HOST,
                 port: int = config.DEFAULT_QDRANT_PORT,
                 collection: str = config.DEFAULT_COLLECTION_NAME,
                 model: str = config.DEFAULT_LLM_MODEL) -> str:
    """
    Process a query using the RAG system:
      - Connect to Qdrant using the given host, port, and collection.
      - Build a retriever.
      - Use the singleton LLM instance (with the provided model) to answer the query.
    """
    embeddings = get_embeddings()
    client = QdrantClient(host=host, port=port)
    qdrant_store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=embeddings
    )
    retriever = qdrant_store.as_retriever(search_kwargs={"k": config.RETRIEVER_K})
    llm = OllamaLLM.get_instance(model)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    qa_response = qa_chain.invoke({"query": query})
    return qa_response["result"]

def build_app():
    with gr.Blocks() as demo:
        gr.Markdown("# CodeBaseRag Interactive UI")
        with gr.Tabs():
            with gr.TabItem("Ask Query"):
                with gr.Column():
                    query_input = gr.Textbox(label="Query", placeholder="Enter your query here")
                    host_input_q = gr.Textbox(label="Qdrant Host", value=config.DEFAULT_QDRANT_HOST)
                    port_input_q = gr.Number(label="Qdrant Port", value=config.DEFAULT_QDRANT_PORT)
                    print("\n----\n")
                    print(list_qdrant_collections())
                    print("\n----\n")
                    collection_dropdown = gr.Dropdown(label="Collection",
                                                      choices=list_qdrant_collections(),
                                                      value=config.DEFAULT_COLLECTION_NAME)
                    model_dropdown = gr.Dropdown(label="LLM Model",
                                                 choices=list_installed_models(),
                                                 value=config.DEFAULT_LLM_MODEL)
                    query_button = gr.Button("Ask Query")
                    # query_output = gr.Textbox(label="Answer")
                    query_output = gr.Code(label="Answer", language="markdown")
                query_button.click(fn=answer_query,
                                   inputs=[query_input, host_input_q, port_input_q, collection_dropdown, model_dropdown],
                                   outputs=query_output)
    return demo

def launch_app():
    app = build_app()
    app.launch(
        share=config.DEFAULT_GRADIO_SHARE,
        server_name=config.DEFAULT_GRADIO_SERVER_NAME,
        server_port=config.DEFAULT_GRADIO_SERVER_PORT
    )

if __name__ == "__main__":
    launch_app()