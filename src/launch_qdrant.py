#!/usr/bin/env python3
import os
import subprocess
import sys
import argparse
from user_interface.config import config

class QdrantManager:
    # Use the configurable container ID file location from the config
    CONTAINER_ID_FILE = config.DEFAULT_CONTAINER_ID_FILE

    @staticmethod
    def launch(port, storage_folder, detach=True):
        # Ensure the designated storage folder exists
        storage_folder = os.path.abspath(storage_folder)
        if not os.path.exists(storage_folder):
            os.makedirs(storage_folder)
            print(f"Created storage folder: {storage_folder}")

        # Build the Docker run command
        command = ["docker", "run"]
        if detach:
            command.append("-d")
        command.extend([
            "-p", f"{port}:6333",
            "-v", f"{storage_folder}:/qdrant/storage",
            "qdrant/qdrant"
        ])

        print("Launching Qdrant with command:")
        print(" ".join(command))

        try:
            if detach:
                # For detached mode, capture the container ID from the output
                result = subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                container_id = result.stdout.strip()
                # Store container ID to a file for later reference
                with open(QdrantManager.CONTAINER_ID_FILE, "w") as f:
                    f.write(container_id)
                print("Qdrant container launched in detached mode. Container ID:", container_id)
                return container_id
            else:
                # For attached mode, do not capture output so logs stream live to the terminal
                subprocess.run(command, check=True)
                print("Qdrant container is running in attached mode.")
        except subprocess.CalledProcessError as e:
            print("Error launching Qdrant container:")
            print(e.stderr)
            sys.exit(1)

    @staticmethod
    def kill():
        # Retrieve the container ID from the file
        if not os.path.exists(QdrantManager.CONTAINER_ID_FILE):
            print("No container ID file found. Is Qdrant running in detached mode?")
            return

        with open(QdrantManager.CONTAINER_ID_FILE, "r") as f:
            container_id = f.read().strip()

        if not container_id:
            print("No container ID found in the file.")
            return

        # Execute the docker kill command
        command = ["docker", "kill", container_id]
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("Successfully killed Qdrant container with ID:", container_id)
            # Remove the file after killing the container
            os.remove(QdrantManager.CONTAINER_ID_FILE)
        except subprocess.CalledProcessError as e:
            print("Error killing Qdrant container:")
            print(e.stderr)
            sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Manage Qdrant Docker container (launch/kill)")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command: launch or kill")

    # Launch subcommand arguments
    launch_parser = subparsers.add_parser("launch", help="Launch Qdrant Docker container")
    launch_parser.add_argument(
        "--port",
        type=int,
        default=config.DEFAULT_QDRANT_PORT,
        help=f"Qdrant host port (default: {config.DEFAULT_QDRANT_PORT})"
    )
    launch_parser.add_argument(
        "--storage-folder",
        type=str,
        default=config.DEFAULT_QDRANT_STORAGE_FOLDER,
        help=f"Path for Qdrant storage folder (default: {config.DEFAULT_QDRANT_STORAGE_FOLDER})"
    )
    launch_parser.add_argument(
        "--detach",
        action="store_true",
        help="Run container in detached mode (default). If not set, container runs in attached mode."
    )

    # Kill subcommand (no additional arguments needed)
    subparsers.add_parser("kill", help="Kill the detached Qdrant Docker container using the stored container ID")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.command == "launch":
        QdrantManager.launch(args.port, args.storage_folder, detach=args.detach)
    elif args.command == "kill":
        QdrantManager.kill()
