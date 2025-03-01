import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

from torchrl.data import LazyTensorStorage, SamplerWithoutReplacement
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.envs.utils import ExplorationType, set_exploration_type

from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from scripts.data import DataCollectorFromEarthGym

class ProximalPolicyOptimization():
    """
    Proximal Policy Optimization (PPO) algorithm.
    """
    def __init__(
            self,
            client: any,
            conf: object,
            policy: nn.Module,
            v_function: nn.Module,
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
        self._v_function = v_function
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
            in_keys=["policy_observation"],
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
            module=self._v_function,
            in_keys=["v_function_observation"]
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

        # Save the logs
        self._pbar.close()
        self.plot_learning_progress()

    def plot_learning_progress(self):
        """
        Plot the learning progress.
        """
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.plot(self._logs["reward"])
        plt.title("Training rewards (average)")
        plt.subplot(2, 2, 2)
        plt.plot(self._logs["step_count"])
        plt.title("Max step count (training)")
        plt.subplot(2, 2, 3)
        plt.plot(self._logs["eval reward (sum)"])
        plt.title("Return (test)")
        plt.subplot(2, 2, 4)
        plt.plot(self._logs["eval step_count"])
        plt.title("Max step count (test)")
        plt.savefig(f"learning_progress.png", dpi=500)
        plt.close()
    
    def test(self, n_steps: int = 10000):
        """
        Run the agent in the environment for a specified number of timesteps.
        """
        self._collector.test(n_steps=n_steps)