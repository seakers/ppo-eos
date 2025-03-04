import json
import sys
import argparse
import traceback

import torch

from scripts.ppo import ProximalPolicyOptimization
from scripts.utils import DataFromJSON
from scripts.client import Client

if __name__ == "__main__":
    try:
        # Check if CUDA is available
        print("CUDA available:", torch.cuda.is_available())

        # Check the GPU name
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Gather the arguments
        argparse = argparse.ArgumentParser()

        argparse.add_argument("--host", default="localhost", type=str, help="Host address.")
        argparse.add_argument("--port", default=5555, type=int, help="Port number.")
        argparse.add_argument("--save", type=str, help="Configuration file.")

        args = argparse.parse_args()

        # Create agent
        client = Client(gym_host=args.host, gym_port=args.port)

        # Load the configuration file
        with open(f"{sys.path[0]}/configuration.json", "r") as file:
            config = json.load(file)

        # Create configuration object
        conf = DataFromJSON(config, "configuration")

        # Create the Proximal Policy Optimization (PPO) object
        ppo = ProximalPolicyOptimization(
            client=client,
            conf=conf,
            save_path=args.save,
            device=device
        )

        # Start PPO
        ppo.start()
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

    finally:
        ppo._client.shutdown_gym()
        