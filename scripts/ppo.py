import torch
import torch.nn as nn
import torch.optim

import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

from torchrl.data import LazyTensorStorage, SamplerWithoutReplacement
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator

from tensordict.nn import TensorDictModule

from scripts.data import DataCollectorFromEarthGym
from scripts.model import SimpleMLP
from scripts.model import MLPModelEOS
from scripts.model import TransformerEncoderModelEOS
from scripts.model import DiscreteStateTransformerEncoderModelEOS
from scripts.model import TransformerModelEOS
from scripts.client import Client
from scripts.utils import DataFromJSON

class ProximalPolicyOptimization():
    """
    Proximal Policy Optimization (PPO).
    """
    def __init__(
            self,
            client: Client,
            conf: DataFromJSON,
            save_path: str = "./",
            device: torch.device = torch.device("cpu")
        ):
        ########################### Parameters ###########################
        self._client = client
        self._conf = conf
        self._save_path = save_path
        self._device = device
        ########################### Parameters ###########################

    def start(self):
        """
        Start the training process.
        """
        # Create the policy model
        policy_net = self.build_policy_net()

        # Create the value model
        value_net = self.build_value_net()

        # Create the PPO algorithm
        ppo_algo = PPOAlgorithm(
            client=self._client,
            conf=self._conf,
            policy=policy_net,
            value_fn=value_net,
            horizon=self._conf.horizon,
            minibatch_size=self._conf.minibatch_size,
            optim_steps=self._conf.optim_steps,
            max_grad=self._conf.max_grad_norm,
            epsilon=self._conf.clip_epsilon,
            gamma=self._conf.discount,
            lmbda=self._conf.gae_lambda,
            c1=self._conf.v_loss_coef,
            c2=self._conf.entropy_coef,
            lr_sched=self._conf.lr_schedule,
            lr_min=self._conf.lr_min,
            device=self._device
        )

        print("Starting the learning process...")

        # Start the PPO algorithm
        ppo_algo.learn(self._conf.learn_steps)

        # Save the model
        ppo_algo.save_models(self._save_path)

        print("Learning process completed. Doing final tests...")

        # Test the model
        test_rewards = ppo_algo.test(self._conf.test_steps)
        ppo_algo._logs["unaveraged final test reward"].extend(test_rewards)

        print(f"Average reward over {self._conf.test_steps} test steps: {sum(test_rewards)/len(test_rewards)}")

        # Plot and save the learning progress
        ppo_algo.plot_learning_progress(self._save_path)

        # Plot the losses
        ppo_algo.plot_losses(self._save_path)

        # Plot time usage
        ppo_algo.plot_time_usage(self._save_path)

    def build_policy_net(self):
        """
        Build the policy model.
        """
        # Add the configuration file properties of the architecture chosen
        for i in range(len(self._conf.archs_available)):
            if self._conf.archs_available[i]["name"] == self._conf.policy_arch:
                policy_conf: defaultdict = self._conf.archs_available[i].copy()
                break

        print(f"Using {policy_conf.pop('name')} architecture for the policy.")

        # Create the policy network
        if self._conf.policy_arch == "SimpleMLP":
            policy_conf["input_dim"] = self._conf.max_len * self._conf.state_dim
            policy_conf["output_dim"] = self._conf.action_dim
            policy_net = SimpleMLP(**policy_conf, device=self._device)
        elif self._conf.policy_arch == "MLP":
            policy_conf["in_dim"] = self._conf.max_len * self._conf.state_dim
            policy_conf["out_dim"] = self._conf.action_dim
            policy_net = MLPModelEOS(**policy_conf, device=self._device)
        elif self._conf.policy_arch == "TransformerEncoder":
            policy_conf["src_dim"] = self._conf.state_dim
            policy_conf["out_dim"] = self._conf.action_dim
            policy_conf["max_len"] = self._conf.max_len
            policy_net = TransformerEncoderModelEOS(**policy_conf, device=self._device)
        elif self._conf.policy_arch == "DiscreteStateTransformerEncoder":
            policy_conf["src_dim"] = self._conf.state_dim
            policy_conf["out_dim"] = self._conf.action_dim
            policy_conf["max_len"] = self._conf.max_len
            policy_net = DiscreteStateTransformerEncoderModelEOS(**policy_conf, device=self._device)
        elif self._conf.policy_arch == "Transformer":
            policy_conf["src_dim"] = self._conf.state_dim
            policy_conf["tgt_dim"] = self._conf.action_dim
            policy_conf["out_dim"] = self._conf.action_dim
            policy_conf["max_len"] = self._conf.max_len
            policy_net = TransformerModelEOS(**policy_conf, device=self._device)
        else:
            raise ValueError(f"Policy architecture {self._conf.policy_arch} not available. Please choose from {[i['name'] for i in self._conf.archs_available]}.")

        if self._conf.load_params and os.path.exists(self._save_path + "/policy.pt"):
            policy_net.load_state_dict(torch.load(self._save_path + "/policy.pt"))
            print("Loaded the policy network.")

        return policy_net.to(self._device)
    
    def build_value_net(self):
        """
        Build the value model.
        """
        # Add the configuration file properties of the architecture chosen
        for i in range(len(self._conf.archs_available)):
            if self._conf.archs_available[i]["name"] == self._conf.value_fn_arch:
                value_conf: defaultdict = self._conf.archs_available[i].copy()
                break

        print(f"Using {value_conf.pop('name')} architecture for the value function.")
        value_conf["is_value_fn"] = True

        # Create the value model
        if self._conf.value_fn_arch == "SimpleMLP":
            value_conf["input_dim"] = self._conf.max_len * self._conf.state_dim
            value_conf["output_dim"] = 1
            value_net = SimpleMLP(**value_conf, device=self._device)
        elif self._conf.policy_arch == "MLP":
            value_conf["in_dim"] = self._conf.max_len * self._conf.state_dim
            value_conf["out_dim"] = 1
            value_net = MLPModelEOS(**value_conf, device=self._device)
        elif self._conf.policy_arch == "TransformerEncoder":
            value_conf["src_dim"] = self._conf.state_dim
            value_conf["out_dim"] = 1
            value_conf["max_len"] = self._conf.max_len
            value_net = TransformerEncoderModelEOS(**value_conf, device=self._device)
        elif self._conf.policy_arch == "DiscreteStateTransformerEncoder":
            value_conf["src_dim"] = self._conf.state_dim
            value_conf["out_dim"] = 1
            value_conf["max_len"] = self._conf.max_len
            value_net = DiscreteStateTransformerEncoderModelEOS(**value_conf, device=self._device)
        elif self._conf.policy_arch == "Transformer":
            value_conf["src_dim"] = self._conf.state_dim
            value_conf["tgt_dim"] = self._conf.action_dim
            value_conf["out_dim"] = 1
            value_conf["max_len"] = self._conf.max_len
            value_net = TransformerModelEOS(**value_conf, device=self._device)
        else:
            raise ValueError(f"Value architecture {self._conf.value_fn_arch} not available. Please choose from {[i['name'] for i in self._conf.archs_available]}.")

        if self._conf.load_params and os.path.exists(self._save_path + "/value_fn.pt"):
            value_net.load_state_dict(torch.load(self._save_path + "/value_fn.pt"))
            print("Loaded the value function network.")

        return value_net.to(self._device)

class PPOAlgorithm():
    """
    Proximal Policy Optimization (PPO) <https://arxiv.org/abs/1707.06347> algorithm.
    """
    def __init__(
            self,
            client: Client,
            conf: object,
            policy: nn.Module,
            value_fn: nn.Module,
            horizon: int = 2048,
            minibatch_size: int = 64,
            optim_steps: int = 10,
            max_grad: float = 1.0,
            epsilon: float = 0.2,
            gamma: float = 0.99,
            lmbda: float = 0.95,
            c1: float = 1.0,
            c2: float = 0.01,
            lr: float = 3e-4,
            lr_sched: bool = False,
            lr_min: float = 0.0,
            device: torch.device = torch.device("cpu")
        ):
        ########################### Parameters ###########################
        # General hyperparameters
        self._client = client
        self._conf = conf
        self._policy = policy
        self._value_fn = value_fn
        self._horizon = horizon
        self._minibatch_size = minibatch_size
        self._optim_steps = optim_steps
        self._max_grad = max_grad
        self._lr_sched = lr_sched
        self._lr_min = lr_min

        # Sensitive hyperparamaters
        self._epsilon = epsilon
        self._gamma = gamma
        self._lmbda = lmbda
        self._c1 = c1
        self._c2 = c2
        self._lr = lr

        # Cuda device
        self._device = device
        ########################### Parameters ###########################

        ########################### PPO Core ###########################
        policy_td_module = TensorDictModule(
            module=self._policy,
            in_keys=["policy_observation"] if self._conf.policy_arch != "Transformer" else ["policy_observation", "actions_as_tgt"],
            out_keys=["loc", "scale"]
        )

        self._actor = ProbabilisticActor(
            module=policy_td_module,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": torch.tensor([-1., -1.], device=self._device), # e.g. tensor([-1., -1., -1.])
                "high": torch.tensor([1., 1.], device=self._device), # e.g. tensor([1., 1., 1.])
            },
            return_log_prob=True
        )

        self._value_module = ValueOperator(
            module=self._value_fn,
            in_keys=["value_fn_observation"] if self._conf.value_fn_arch != "Transformer" else ["value_fn_observation", "actions_as_tgt"]
        )

        self._advantage_module = GAE(
            gamma=self._gamma,
            lmbda=self._lmbda,
            value_network=self._value_module,
            average_gae=True
        )

        self._replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self._horizon),
            sampler=SamplerWithoutReplacement(),
        )

        self._loss_module = ClipPPOLoss(
            actor_network=self._actor,
            critic_network=self._value_module,
            clip_epsilon=(self._epsilon),
            entropy_bonus=True,
            critic_coef=self._c1,
            entropy_coef=self._c2
        )
        ########################### PPO Core ###########################  

    def learn(self, total_steps: int=10000):
        """
        Learn the policy using PPO. Inspired from the torchrl implementation.
        """
        ########################### Learning ###########################
        self._total_steps = total_steps

        self._collector = DataCollectorFromEarthGym(
            client=self._client,
            conf=self._conf,
            policy=self._actor,
            batch_steps=self._horizon,
            total_steps=self._total_steps,
            device=self._device
        )

        self._optimizer = torch.optim.Adam(self._loss_module.parameters(), self._lr)
        self._lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer, self._total_steps // self._horizon, self._lr_min if self._lr_sched else self._lr
        )
        ########################### Learning ###########################

        ########################### Tracking ###########################
        self._logs = defaultdict(list)
        self._pbar = tqdm(total=self._total_steps)
        ########################### Tracking ###########################

        self._actor.train()
        for i, tensordict_data in enumerate(self._collector):
            start_time = datetime.now()

            # We now have a batch of data in tensordict_data
            objective_losses = []
            critic_losses = []
            entropy_losses = []

            if self._conf.debug:
                advantage_times = []
                loss_times = []
                backprop_times = []

            for _ in range(self._optim_steps):
                # We'll need an "advantage" signal to make PPO work
                # Compute GAE at each epoch as its value depends on the updated value function
                if self._conf.debug:
                    now = datetime.now()

                self._advantage_module(tensordict_data.to(self._device))
                self._replay_buffer.extend(tensordict_data.cpu())

                if self._conf.debug:
                    advantage_times.append((datetime.now() - now).total_seconds())

                for _ in range(self._horizon // self._minibatch_size):
                    if self._conf.debug:
                        now = datetime.now()

                    minibatch = self._replay_buffer.sample(self._minibatch_size)
                    loss_vals = self._loss_module(minibatch.to(self._device))
                    loss_value: torch.Tensor = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    if self._conf.debug:
                        loss_times.append((datetime.now() - now).total_seconds())
                        now = datetime.now()

                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self._loss_module.parameters(), self._max_grad)
                    self._optimizer.step()
                    self._optimizer.zero_grad()

                    if self._conf.debug:
                        backprop_times.append((datetime.now() - now).total_seconds())

                    # Append loss values
                    objective_losses.append(loss_vals["loss_objective"].item())
                    critic_losses.append(loss_vals["loss_critic"].item())
                    entropy_losses.append(loss_vals["loss_entropy"].item())

            ########################### Testing ###########################
            if i % 10 == 0:
                # Test the model
                test_rewards = self.test(n_steps=100)
                self._collector.switch_trajectory()
                self._logs["unaveraged test reward"].extend(test_rewards)
                self._logs["test reward"].append(sum(test_rewards)/len(test_rewards))
            ########################### Testing ###########################

            # self._logs["unaveraged reward"].append(tensordict_data["next", "reward"])
            self._logs["reward"].append(tensordict_data["next", "reward"].mean().item())
            reward_str = f"Avg rewards = (test: {self._logs['test reward'][-1]:6f} | train: {self._logs['reward'][-1]:6f}), "
            self._logs["loss_objective"].append(sum(objective_losses)/len(objective_losses))
            self._logs["loss_critic"].append(sum(critic_losses)/len(critic_losses))
            self._logs["loss_entropy"].append(sum(entropy_losses)/len(entropy_losses))
            losses_str = f"Losses = (objective: {self._logs['loss_objective'][-1]:.6f} | critic: {self._logs['loss_critic'][-1]:.6f} | entropy: {self._logs['loss_entropy'][-1]:.6f}), "
            self._logs["lr"].append(self._optimizer.param_groups[0]["lr"])
            lr_str = f"Learning rate: {self._logs['lr'][-1]:.6f}"
            
            # Update the progress bar
            self._pbar.update(tensordict_data.numel())
            self._pbar.set_description("".join([reward_str, losses_str, lr_str]))

            # Update the learning rate
            self._lr_scheduler.step()

            self._logs["time"].append((datetime.now() - start_time).total_seconds())

            if self._conf.debug:
                print()
                print(f"""Cumulative time taken for optimization: {self._logs['time'][-1]:.4f}
    - Advantage: {sum(advantage_times):.4f}s
    - Optim: {sum(loss_times):.4f}s
    - Backprop: {sum(backprop_times):.4f}s""")

        # Save the logs
        self._pbar.close()
    
    def test(self, n_steps: int=10000):
        """
        Run the agent in the environment for a specified number of timesteps.
        """
        return self._collector.test(n_steps=n_steps)

    def plot_learning_progress(self, path: str="."):
        """
        Plot the learning progress.
        """
        # Plot training rewards smoothed
        rewards_df = pd.DataFrame(self._logs["reward"], columns=["Reward"])
        rewards_df["Reward (smoothed)"] = rewards_df["Reward"].rolling(window=int(len(rewards_df["Reward"])/10)).mean()

        plt.plot(rewards_df["Reward (smoothed)"])
        plt.title("Training rewards (average)")
        plt.xlabel(f"Experience batches ({self._horizon} steps each)")
        plt.ylabel("Reward")
        plt.savefig(f"{path}/training_progress.png", dpi=500)
        plt.close()

        # Plot testing rewards smoothed
        learning_test_size = len(self._logs["unaveraged test reward"])
        self._logs["unaveraged test reward"].extend(self._logs["unaveraged final test reward"])
        rewards_df = pd.DataFrame(self._logs["unaveraged test reward"], columns=["Reward"])
        rewards_df["Reward (smoothed)"] = rewards_df["Reward"].rolling(window=int(len(rewards_df["Reward"])/10)).mean()

        plt.plot(rewards_df["Reward (smoothed)"])
        plt.axvline(x=learning_test_size, color="red", linestyle="--", linewidth=1)
        plt.title("Testing rewards (average)")
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.savefig(f"{path}/testing_progress.png", dpi=500)
        plt.close()

    def plot_losses(self, path: str="."):
        """
        Plot the losses.
        """
        # Plot losses
        losses_df = pd.DataFrame(self._logs["loss_objective"], columns=["Objective"])
        losses_df["Critic"] = self._logs["loss_critic"]
        losses_df["Entropy"] = self._logs["loss_entropy"]

        losses_df["Objective"] = losses_df["Objective"].rolling(window=int(len(losses_df["Objective"])/10)).mean()
        losses_df["Critic"] = losses_df["Critic"].rolling(window=int(len(losses_df["Critic"])/10)).mean()
        losses_df["Entropy"] = losses_df["Entropy"].rolling(window=int(len(losses_df["Entropy"])/10)).mean()

        plt.plot(losses_df["Objective"], label="Objective")
        plt.plot(losses_df["Critic"], label="Critic")
        plt.plot(losses_df["Entropy"], label="Entropy")
        plt.title("Losses")
        plt.legend()
        plt.savefig(f"{path}/losses.png", dpi=500)
        plt.close()

    def plot_time_usage(self, path: str="."):
        """
        Plot the time usage.
        """
        # Plot time usage
        time_df = pd.DataFrame(self._logs["time"], columns=["Time"])
        time_df["Time (smoothed)"] = time_df["Time"].rolling(window=int(len(time_df["Time"])/10)).mean()

        plt.plot(time_df["Time (smoothed)"])
        plt.title("Time usage per horizon")
        plt.savefig(f"{path}/time_usage.png", dpi=500)
        plt.close()

    def save_models(self, path: str = "."):
        """
        Save the following models:
        - Policy
        - Probabilistic actor
        - Value function
        - Value operator 
        """
        # Make sure the path exists
        if not os.path.exists(path):
            os.makedirs(path)

        # Save in .pt files
        torch.save(self._policy.state_dict(), f"{path}/policy.pt")
        torch.save(self._actor.state_dict(), f"{path}/prob_actor.pt")
        torch.save(self._value_fn.state_dict(), f"{path}/value_fn.pt")
        torch.save(self._value_module.state_dict(), f"{path}/value_operator.pt")