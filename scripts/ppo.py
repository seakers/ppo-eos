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
            device=self._device
        )

        print("Starting the learning process...")

        # Start the PPO algorithm
        ppo_algo.learn(self._conf.learn_steps)

        # Test the model
        ppo_algo.test(self._conf.test_steps)

        # Save the model
        ppo_algo.save_models(self._save_path)

        # Plot and save the learning progress
        ppo_algo.plot_learning_progress(self._save_path)

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
        elif self._conf.policy_arch == "Transformer":
            policy_conf["src_dim"] = self._conf.state_dim
            policy_conf["tgt_dim"] = self._conf.action_dim
            policy_conf["out_dim"] = self._conf.action_dim
            policy_conf["max_len"] = self._conf.max_len
            policy_net = TransformerModelEOS(**policy_conf, device=self._device)
        else:
            raise ValueError(f"Policy architecture {self._conf.policy_arch} not available. Please choose from {[i["name"] for i in self._conf.archs_available]}.")

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
        elif self._conf.policy_arch == "Transformer":
            value_conf["src_dim"] = self._conf.state_dim
            value_conf["tgt_dim"] = self._conf.action_dim
            value_conf["out_dim"] = 1
            value_conf["max_len"] = self._conf.max_len
            value_net = TransformerModelEOS(**value_conf, device=self._device)
        else:
            raise ValueError(f"Value architecture {self._conf.value_fn_arch} not available. Please choose from {[i['name'] for i in self._conf.archs_available]}.")

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
            lr_sched: bool = True,
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
                "low": torch.tensor([-1., -1.]), # e.g. tensor([-1., -1., -1.])
                "high": torch.tensor([1., 1.]), # e.g. tensor([1., 1., 1.])
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

    def learn(self, total_steps: int = 10000):
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

        eval_str = ""
        self._actor.train()
        for i, tensordict_data in enumerate(self._collector):
            now = datetime.now()
            # We now have a batch of data in tensordict_data
            for _ in range(self._optim_steps):
                # We'll need an "advantage" signal to make PPO work
                # Compute GAE at each epoch as its value depends on the updated value function
                self._advantage_module(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                self._replay_buffer.extend(data_view.cpu())

                for _ in range(self._horizon // self._minibatch_size):
                    minibatch = self._replay_buffer.sample(self._minibatch_size)
                    loss_vals = self._loss_module(minibatch.to(self._device))
                    loss_value: torch.Tensor = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self._loss_module.parameters(), self._max_grad)
                    self._optimizer.step()
                    self._optimizer.zero_grad()

            self._logs["reward"].append(tensordict_data["next", "reward"].mean().item())
            cum_reward_str = (
                f"Average reward: {self._logs["reward"][-1]:4f}, "
            )
            self._logs["step_count"].append(tensordict_data["next", "step_count"].max().item())
            stepcount_str = f"Step count (max): {self._logs["step_count"][-1]}, "
            self._logs["lr"].append(self._optimizer.param_groups[0]["lr"])
            lr_str = f"Policy lr: {self._logs["lr"][-1]:.6f}"

            ########################### Testing ###########################
            # if i % 10 == 0 and False:
            #     rewards, step_count = self.test_one_run()
            #     self._actor.train()
            #     self._logs["eval reward (sum)"].append(rewards)
            #     self._logs["eval step_count"].append(step_count)

            #     self._logs["reward"].append(tensordict_data["next", "reward"].mean().item())
            #     self._logs["step_count"].append(tensordict_data["step_count"].max().item())
            #     self._logs["lr"].append(self._optimizer.param_groups[0]["lr"])
            #     eval_str = (
            #         f"Eval cumulative reward: {self._logs["eval reward (sum)"][-1]:4f}, "
            #         f"Eval step count: {self._logs["eval step_count"][-1]}, "
            #     )
            ########################### Testing ###########################
            
            # Update the progress bar
            self._pbar.update(tensordict_data.numel())
            self._pbar.set_description("".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
            # self._pbar.set_description(f"Current test rewards is {rewards:.4f} and step count is {step_count:.0f}. Learning rate is {last_lr[0]:.6f}")

            # Update the learning rate
            self._lr_scheduler.step()

            self._logs["time"].append((datetime.now() - now).total_seconds())

        # Save the logs
        self._pbar.close()
    
    def test(self, n_steps: int = 10000):
        """
        Run the agent in the environment for a specified number of timesteps.
        """
        print("Testing the model...")
        self._collector.test(n_steps=n_steps)

    def plot_learning_progress(self, path: str = "."):
        """
        Plot the learning progress.
        """
        # Plot rewards smoothed and with shaded error
        rewards_df = pd.DataFrame(self._logs["reward"], columns=["Reward"])
        rewards_df["Reward (smoothed)"] = rewards_df["Reward"].rolling(window=int(len(rewards_df["Reward"])/10)).mean()

        plt.plot(rewards_df["Reward (smoothed)"])
        plt.title("Training rewards (average)")
        plt.savefig(f"{path}/learning_progress.png", dpi=500)
        plt.close()

    def plot_time_usage(self, path: str = "."):
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