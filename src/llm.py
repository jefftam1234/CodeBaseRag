#!/usr/bin/env python3
import subprocess
from langchain.llms.base import LLM
from typing import Optional, List
import argparse
from user_interface.config import config
from pydantic import Field, model_validator

class OllamaLLM(LLM):
    # Declare the model field so Pydantic knows about it.
    model: str = Field(..., description="The Ollama model to use.")

    # A class-level flag to control printing (default False means print)
    _suppress_print: bool = False

    @model_validator(mode="after")
    def print_model(self) -> "OllamaLLM":
        if not getattr(self.__class__, "_suppress_print", False):
            print(f"OllamaLLM instance created with model: {self.model}")
        return self

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        result = subprocess.run(
            ["ollama", "run", self.model],
            input=prompt,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()

    @property
    def _identifying_params(self):
        return {"model": self.model}

    @property
    def _llm_type(self) -> str:
        return "ollama"

    @classmethod
    def get_instance(cls, model: Optional[str] = None, verbose: bool = True):
        """
        Returns the singleton instance of OllamaLLM.
        If no instance exists, it is created using the provided model
        (or the default from config).
        The `verbose` flag controls whether the model is printed on instantiation.
        """
        if not hasattr(cls, "_instance") or cls._instance is None:
            if model is None:
                model = config.DEFAULT_LLM_MODEL
            # Set the class-level flag based on verbose
            cls._suppress_print = not verbose
            cls._instance = cls(model=model)
            # Optionally reset the flag after instantiation if you want future calls to print.
            # For example:
            # cls._suppress_print = False
        return cls._instance

    @classmethod
    def get_llm(cls, model: Optional[str] = None, verbose: bool = True):
        return cls.get_instance(model, verbose)

    @staticmethod
    def list_installed_llms():
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode != 0:
            print("Error listing installed LLM models:")
            print(result.stderr)
            return
        print("Installed LLM models:")
        print(result.stdout)
        print("\nUse one of the above model names in your config file as DEFAULT_LLM_MODEL (or override with --model).")

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
        # Using invoke() per the new LangChain API
        response = llm.invoke(args.prompt)
        print("Response:", response)
    else:
        parser.print_help()
