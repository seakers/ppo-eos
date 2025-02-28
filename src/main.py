import json
import sys
import argparse
import traceback

import torch
import torch.nn as nn
from tensordict.nn.distributions import NormalParamExtractor

from scripts.ppo import ProximalPolicyOptimization
from scripts.model import SimpleMLP
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

        # Create the policy model
        n_hidden = 256
        policy_net = SimpleMLP(
            input_dim=conf.max_len*conf.state_dim,
            n_hidden=n_hidden,
            output_dim=conf.action_dim,
            device=device
        )

        # Create the value model
        value_net = SimpleMLP(
            input_dim=conf.max_len*conf.state_dim,
            n_hidden=n_hidden,
            output_dim=1,
            device=device
        )

        # Create the SAC algorithm
        ppo = ProximalPolicyOptimization(
            client=client,
            conf=conf,
            policy=policy_net,
            v_function=value_net,
            horizon=conf.horizon,
            minibatch_size=conf.minibatch_size,
            optim_steps=conf.optim_steps,
            max_grad=conf.max_grad_norm,
            epsilon=conf.clip_epsilon,
            gamma=conf.discount,
            lmbda=conf.gae_lambda,
            c1=conf.v_loss_coef,
            c2=conf.entropy_coef,
            device=device
        )

        # Start the SAC algorithm
        ppo.learn(conf.learn_steps)

        # Test the model
        ppo.test(conf.test_steps)
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

    finally:
        ppo._collector._client.shutdown_gym()
        