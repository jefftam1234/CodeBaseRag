#!/usr/bin/env python3
import subprocess
import os
from user_interface.config import config, load_config_from_ini, overwrite_config_ini, display_current_config

def prepare_codebase():
    """
    Run the data preparation steps:
    1. Convert source files to text.
    2. Load the converted text and pickle documents.
    3. Split the documents into chunks.
    """
    load_config_from_ini()
    try:
        print("\n--- Preparing Codebase ---")
        print("Running conversion...")
        subprocess.run(["python", "src/convert.py", "--src", config.DEFAULT_CODEBASE_PATH, "--dst", config.DEFAULT_CONVERTED_PATH], check=True)
        print("Running loader...")
        subprocess.run(["python", "src/loader.py", "--src", config.DEFAULT_CONVERTED_PATH, "--dst", config.DEFAULT_DOCS_PICKLE], check=True)
        print("Running splitter...")
        subprocess.run(["python", "src/splitter.py", "--input", config.DEFAULT_DOCS_PICKLE, "--output", config.DEFAULT_CHUNKS_PICKLE], check=True)
        print("Preparation complete.\n")
    except subprocess.CalledProcessError as e:
        print("An error occurred during preparation:", e)
        return

def push_to_qdrant():
    """
    Push the document chunks (pickle file) to Qdrant.
    """
    try:
        print("\n--- Pushing to Qdrant ---")
        subprocess.run(["python", "src/push_to_qdrant.py", config.DEFAULT_CHUNKS_PICKLE, "--collection_name", config.DEFAULT_COLLECTION_NAME], check=True)
        print("Documents successfully pushed to Qdrant.\n")
    except subprocess.CalledProcessError as e:
        print("An error occurred while pushing documents to Qdrant:", e)
        return

def launch_qdrant(launch=True):
    """
    Launch or kill Qdrant using the dedicated script.
    If launch=True, run in detached mode (default). Otherwise, kill the container.
    """
    script_path = os.path.join("src", "launch_qdrant.py")
    try:
        if launch:
            subprocess.run(["python", script_path, "launch", "--detach"], check=True)
            print("Qdrant launched in detached mode.\n")
        else:
            subprocess.run(["python", script_path, "kill"], check=True)
            print("Qdrant container killed.\n")
    except subprocess.CalledProcessError as e:
        print("An error occurred while managing Qdrant:", e)
        return

def launch_cli():
    """
    Launch an interactive CLI interface.
    The user can repeatedly enter queries until they type '/exit'.
    """
    print("\n--- Entering CLI Mode ---")
    print("Type '/exit' to return to the main menu.\n")
    # Import the query_index function (assuming defaults from config)
    from user_interface.cli import query
    while True:
        prompt = input("Enter your query: ").strip()
        if prompt == "/exit":
            print("Exiting CLI mode...\n")
            break
        try:
            answer = query(prompt,
                           config.DEFAULT_QDRANT_HOST,
                           config.DEFAULT_QDRANT_PORT,
                           config.DEFAULT_COLLECTION_NAME,
                           config.DEFAULT_LLM_MODEL,
                           suppress_output=True)
            print("Answer:", answer, "\n")
            print("Type '/exit' to return to the main menu.\n")
        except Exception as e:
            print("Error processing query:", e)

def launch_gui():
    """
    Launch the Gradio GUI in a background process.
    Wait for the user to press Enter to kill the GUI and return to the main menu.
    """
    try:
        print("\n--- Launching Gradio GUI ---")
        # Launch in background so that the interactive menu is not blocked.
        proc = subprocess.Popen(["python", "user_interface/gradio_app.py"])
        print(f"Gradio GUI launched with PID: {proc.pid}")
        input("Press Enter to terminate the GUI and return to the main menu...")
        proc.terminate()
        proc.wait()
        print("Gradio GUI terminated.\n")
    except Exception as e:
        print("An error occurred while launching the Gradio UI:", e)

def interactive_menu():
    while True:
        print("\n=== CodeBaseRag Main Menu ===")
        print("1. Prepare Codebase (Convert, Load, Split)")
        print("2. Manage Qdrant (Launch/Kill)")
        print("3. Push to Qdrant")
        print("4. Use Interface (CLI/GUI)")
        print("9. Display current config/Reload config from file")
        print("0. Exit")
        choice = input("Select an option: ").strip()

        if choice == "1":
            prepare_codebase()
        elif choice == "2":
            sub_choice = input("Enter 'l' to launch Qdrant or 'k' to kill it: ").strip().lower()
            if sub_choice == "l":
                launch_qdrant(launch=True)
            elif sub_choice == "k":
                launch_qdrant(launch=False)
            else:
                print("Invalid option for Qdrant management.")
        elif choice == "3":
            push_to_qdrant()
        elif choice == "4":
            sub_choice = input("Enter 'c' for CLI or 'g' for GUI: ").strip().lower()
            if sub_choice == "c":
                launch_cli()
            elif sub_choice == "g":
                launch_gui()
            else:
                print("Invalid option for interface selection.")
        elif choice == "9":
            sub_choice = input("Enter 'd' to display current config, or 'r' to reload config from another file: ").strip().lower()
            if sub_choice == "d":
                display_current_config()
            elif sub_choice == "r":
                new_config_file = input("Enter the path to the new config file: ").strip()
                try:
                    overwrite_config_ini(new_config_file)
                    print(f"Config reloaded from {new_config_file}.")
                    print("Exiting interactive menu. Please re-launch the application for updated configuration.")
                    break  # Exit the interactive menu so that on next launch, the new config is loaded.
                except Exception as e:
                    print(f"Error reloading config from {new_config_file}:", e)
            else:
                print("Invalid option for interface selection.")
        elif choice == "0":
            print("Exiting CodeBaseRag. Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")

def main():
    interactive_menu()

if __name__ == "__main__":
    main()