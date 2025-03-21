import io
import os
import sys
import json
import psutil
import pstats
import cProfile
import argparse
import traceback
import tracemalloc

import torch

from scripts.ppo import ProximalPolicyOptimization
from scripts.utils import DataFromJSON
from scripts.client import Client

if __name__ == "__main__":
    try:
        print("Starting the main script of ppo-eos...")

        # Gather the arguments
        argparse = argparse.ArgumentParser()

        argparse.add_argument("--host", default="localhost", type=str, help="Host address.")
        argparse.add_argument("--port", default=5555, type=int, help="Port number.")
        argparse.add_argument("--save", type=str, help="Configuration file.")
        argparse.add_argument("--pro", type=int, help="Profiling and memory allocation mode.")

        args = argparse.parse_args()

        if args.pro:
            print("Tracking profile and memory allocation...")
            # Start tracing memory allocations
            tracemalloc.start()

        # Check if CUDA is available
        print("CUDA available:", torch.cuda.is_available())

        # Check the GPU name
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        if args.pro:
            ###################### PPO with profiling ######################
            # Check the performance
            pr = cProfile.Profile()
            pr.enable()

            # Start PPO
            ppo.start() # cProfile.runctx("ppo.start()", globals(), locals())

            pr.disable()

            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats("tottime") # sort by total time
            ps.print_stats(20) # show top n slowest functions
            print(s.getvalue())
            pr.dump_stats("src/main-profile.prof")
            ###################### PPO with profiling ######################
        else:
            # Start PPO
            ppo.start()
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

    finally:
        if args.pro:
            ###################### Memory allocation ######################
            # Take a snapshot after executing the code
            snapshot = tracemalloc.take_snapshot()

            # Get the top memory allocations
            top_stats = snapshot.statistics('lineno')

            print("Top 100 memory-consuming lines:")
            for stat in top_stats[:100]:
                print(stat)

            # Stop tracing
            tracemalloc.stop()
            ###################### Memory allocation ######################
        
        # Memory used
        process = psutil.Process(os.getpid())
        memory_used = process.memory_info().rss
        print(f"Memory used: {memory_used / (1024 ** 2):.2f} MB")

        # Shutdown the gym
        ppo._client.shutdown_gym()
        