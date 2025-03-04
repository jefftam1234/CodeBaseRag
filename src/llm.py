#!/usr/bin/env python3
import argparse
import atexit
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import requests
import subprocess
import time
import signal
import psutil
from typing import Optional, List

from langchain.llms.base import LLM
from pydantic import Field, PrivateAttr, model_validator
from user_interface.config import config
import ollama



def kill_process_tree(pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    # Recursively get all children
    children = parent.children(recursive=True)
    for child in children:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        parent.kill()
    except psutil.NoSuchProcess:
        pass


class OllamaLLM(LLM):
    # The model field is used by Pydantic for validation.
    model: str = Field(..., description="The Ollama model to use.")
    # Class-level flag to control printing (False means print by default)
    _suppress_print: bool = False
    # Private attribute to hold the server process
    _server_process: Optional[subprocess.Popen] = PrivateAttr(None)

    @model_validator(mode="after")
    def print_model(self) -> "OllamaLLM":
        if not getattr(self.__class__, "_suppress_print", False):
            print(f"OllamaLLM instance created with model: {self.model}")
        return self

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = ollama.generate(model=self.model, prompt=prompt)
        return response.get('response', '').strip()

    @property
    def _identifying_params(self):
        return {"model": self.model}

    @property
    def _llm_type(self) -> str:
        return "ollama"

    @classmethod
    def get_instance(cls, model: Optional[str] = None, verbose: bool = True, start_server: bool = True):
        """
        Returns the singleton instance of OllamaLLM.
        If no instance exists, it is created using the provided model (or the default from config).
        If start_server is True, it will start the Ollama server in a separate process group.
        """
        if not hasattr(cls, "_instance") or cls._instance is None:
            if model is None:
                model = config.DEFAULT_LLM_MODEL
            cls._suppress_print = not verbose
            instance = cls(model=model)
            if start_server:
                print("Starting Ollama server...")
                instance._server_process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid  # Start in a new process group
                )
                # Allow the server time to start
                time.sleep(5)
                # Register cleanup handler to terminate the process group on exit.
                atexit.register(cls.cleanup_instance)
            cls._instance = instance
        return cls._instance

    @classmethod
    def get_llm(cls, model: Optional[str] = None, verbose: bool = True, start_server: bool = True):
        return cls.get_instance(model=model, verbose=verbose, start_server=start_server)

    def unload_model(self) -> bool:
        """
        Instructs the Ollama server to unload this model by setting keep_alive to 0.
        Returns True if the request was successful.
        """
        host_url = config.DEFAULT_OLLAMA_HOST
        url = f"{host_url}/api/generate"
        payload = {"model": self.model, "keep_alive": 0}
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            print(f"Model {self.model} unloaded successfully.")
            return True
        except requests.RequestException as e:
            print("Error unloading model:", e)
            return False

    @classmethod
    def cleanup_instance(cls):
        """
        First sends a request to unload the model (freeing VRAM), then cleans up the server process.
        """
        if hasattr(cls, "_instance") and cls._instance is not None:
            instance = cls._instance
            # Unload the model from Ollama
            instance.unload_model()
            # Now clean up the server process (using psutil-based cleanup as needed)
            if instance._server_process is not None:
                print("Recursively terminating Ollama server process tree...")
                kill_process_tree(instance._server_process.pid, sig=signal.SIGTERM)
                try:
                    instance._server_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    kill_process_tree(instance._server_process.pid, sig=signal.SIGKILL)
                instance._server_process = None
            cls._instance = None

    @staticmethod
    def list_installed_llms():
        result = ollama.list()
        print("Installed LLM models:")
        print(result)
        print("\nUse one of the above model names in your config file as DEFAULT_LLM_MODEL (or override with --model).")

# Example usage as a standalone script:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ollama LLM Helper (Singleton)")
    parser.add_argument("--list", action="store_true", help="List installed LLM models.")
    parser.add_argument("--prompt", type=str, help="A prompt to send to the LLM.")
    parser.add_argument("--model", type=str, help="Specify the model to use (overrides config).")
    parser.add_argument("--quiet", action="store_true", help="Suppress model printing on instantiation.")
    args = parser.parse_args()

    if args.list:
        OllamaLLM.list_installed_llms()
    elif args.prompt:
        llm = OllamaLLM.get_llm(args.model, verbose=not args.quiet)
        response = llm.invoke(args.prompt)
        print("Response:", response)
    else:
        parser.print_help()
    OllamaLLM.cleanup_instance()