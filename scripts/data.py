import torch
import numpy as np

from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor

from scripts.client import Client

RT = 6371.0 # Earth radius in km

class DataCollectorFromEarthGym():
    def __init__(
            self,
            client: Client,
            conf: object,
            policy: ProbabilisticActor,
            batch_steps: int,
            total_steps: int,
            device: torch.device = torch.device("cpu")
        ):
        # Init variables
        self._client = client
        self._conf = conf
        self._policy = policy
        self._batch_steps = batch_steps
        self._total_steps = total_steps
        self._device = device

        # Additional variables
        self._current_step = 0
        self._traj_id = 0
        self._step_count = 0

        self._conf.trajectory_len = float(self._conf.trajectory_len)

        # Warm-up the agent in the environment
        self.initialize_env()

    def __iter__(self):
        """
        Returns the iterator object.
        """
        return self
    
    def __next__(self):
        """
        Collects data from the environment using the policy.
        """
        # Fields from main td
        _action = []
        _done = []
        _actions_as_tgt = []
        _loc = []
        _policy_observation = []
        _value_fn_observation = []
        _sample_log_prob = []
        _scale = []
        _step_count = []
        _terminated = []
        _truncated = []
        
        # Fields from collector td
        _collector_traj_id = []

        # Fields from next td
        _next_done = []
        _next_policy_observation = []
        _next_value_fn_observation = []
        _next_reward = []
        _next_step_count = []
        _next_terminated = []
        _next_truncated = []

        if self._current_step < self._total_steps:
            for _ in range(self._batch_steps):
                with set_exploration_type(ExplorationType.RANDOM):
                    observation, _ = self.prettify_observation(self._states, self._actions)
                    actions_as_tgt = self._actions.clone()
                    loc, scale, action, log_prob = self._policy(observation) if self._conf.policy_arch != "Transformer" else self._policy(observation, actions_as_tgt)
                    curr_policy_obs, curr_value_fn_obs, next_policy_obs, next_value_fn_obs, reward, done = self.move_once(action)

                if done:
                    raise StopIteration
                
                if self._step_count >= self._conf.trajectory_len:
                    terminated = truncated = True
                else:
                    terminated = truncated = False

                # In transformer architectures, we need to ignore the batch dimension (they will stack)
                while observation.dim() > 2:
                    observation = observation.squeeze(0)
                    curr_policy_obs = curr_policy_obs.squeeze(0)
                    curr_value_fn_obs = curr_value_fn_obs.squeeze(0)
                    next_policy_obs = next_policy_obs.squeeze(0)
                    next_value_fn_obs = next_value_fn_obs.squeeze(0)
                    actions_as_tgt = actions_as_tgt.squeeze(0)

                # In transformer architectures, we have ignored all actions except the last one so we need to ignore other dimensions (they will stack)
                while action.dim() > 1:
                    action = action.squeeze(0)
                    loc = loc.squeeze(0)
                    scale = scale.squeeze(0)

                # Fields from main td
                _action.append(action)
                _done.append(False)
                _actions_as_tgt.append(actions_as_tgt)
                _loc.append(loc)
                _policy_observation.append(curr_policy_obs)
                _value_fn_observation.append(curr_value_fn_obs)
                _sample_log_prob.append(log_prob)
                _scale.append(scale)
                _step_count.append(self._step_count)
                _terminated.append(False)
                _truncated.append(False)

                # Fields from collector td
                _collector_traj_id.append(self._traj_id)

                # Fields from next td
                _next_done.append(terminated or truncated)
                _next_policy_observation.append(next_policy_obs)
                _next_value_fn_observation.append(next_value_fn_obs)
                _next_reward.append(reward)
                _next_step_count.append(self._step_count + 1)
                _next_terminated.append(terminated)
                _next_truncated.append(truncated)

                if self._step_count >= self._conf.trajectory_len:
                    self.switch_trajectory()
                else:
                    self._step_count += 1

                self._current_step += 1

            return TensorDict({
                "action": torch.stack(_action, dim=0).clone().detach().to(self._device),
                "actions_as_tgt": torch.stack(_actions_as_tgt, dim=0).clone().detach().to(self._device),
                "done": torch.tensor(_done, device=self._device),
                "loc": torch.stack(_loc, dim=0).clone().detach().to(self._device),
                "policy_observation": torch.stack(_policy_observation, dim=0).clone().detach().to(self._device),
                "value_fn_observation": torch.stack(_value_fn_observation, dim=0).clone().detach().to(self._device),
                "sample_log_prob": torch.stack(_sample_log_prob, dim=0).clone().detach().to(self._device),
                "scale": torch.stack(_scale, dim=0).clone().detach().to(self._device),
                "step_count": torch.tensor(_step_count, device=self._device),
                "terminated": torch.tensor(_terminated, device=self._device),
                "truncated": torch.tensor(_truncated, device=self._device),
                
                "collector": TensorDict({
                    "traj_ids": torch.tensor(_collector_traj_id, device=self._device)
                }, batch_size=self._batch_steps),

                "next": TensorDict({
                    "actions_as_tgt": torch.stack(_actions_as_tgt, dim=0).clone().detach().to(self._device),
                    "done": torch.tensor(_next_done, device=self._device),
                    "policy_observation": torch.stack(_next_policy_observation, dim=0).clone().detach().to(self._device),
                    "value_fn_observation": torch.stack(_next_value_fn_observation, dim=0).clone().detach().to(self._device),
                    "reward": torch.stack(_next_reward, dim=0).clone().detach().to(self._device),
                    "step_count": torch.tensor(_next_step_count, device=self._device),
                    "terminated": torch.tensor(_next_terminated, device=self._device),
                    "truncated": torch.tensor(_next_truncated, device=self._device),
                }, batch_size=self._batch_steps),
            }, batch_size=self._batch_steps)
        else:
            raise StopIteration
    
    def initialize_env(self):
        """
        Initialize the environment. Make the agent do a dummy move to stabilize the observation state.
        """
        sending_data = {
            "agent_id": 0,
            "action": {
                "d_pitch": 0,
                "d_roll": 0
            },
            "delta_time": 0
        }
        state, _, _ = self._client.get_next_state("get_next", sending_data)

        # Normalize the state given by the environment
        vec_state = self.normalize_state(state)

        # Input tensor of 1 batch and 1 sequence of state_dim dimensional states
        self._states = torch.tensor([[vec_state]], dtype=torch.float32, device=self._device)

        # Input tensor of 1 batch and 1 sequence of action_dim dimensional actions
        self._actions = torch.tensor([[[0] * self._conf.action_dim]], dtype=torch.float, device=self._device)

        # Make max_len dummy moves to have a long enough observation
        self.n_dummy_moves(n=self._conf.max_len)

    def switch_trajectory(self):
        """
        Start a new trajectory.
        """
        self._traj_id += 1
        self._step_count = 0

    def n_dummy_moves(self, n: int):
        """
        Do n dummy moves to stabilize the environment.
        """
        for _ in range(n):
            _, _, _, _, _, _ = self.move_once(torch.tensor([0] * self._conf.action_dim, dtype=torch.float32))

    def move_once(self, action: torch.Tensor):
        """
        Do an environment step for the SAC algorithm.
        """
        with torch.no_grad():
            # Get the current observation
            curr_policy_obs, curr_value_fn_obs = self.prettify_observation(self._states, self._actions)

            # --------------- Environment's job to provide info ---------------
            sending_data = {
                "agent_id": 0,
                "action": {
                    "d_pitch": action[(-1,) * (action.dim() - 1) + (0,)].item() * self._conf.a_conversions[0],
                    "d_roll": action[(-1,) * (action.dim() - 1) + (1,)].item() * self._conf.a_conversions[1]
                },
                "delta_time": self._conf.time_increment
            }
            
            state, reward, done = self._client.get_next_state("get_next", sending_data)

            # Break if time is up
            if done:
                print("Time is up!")
                return None, None, True

            # Normalize the state
            vec_state = self.normalize_state(state)

            # Get the reward
            r = torch.tensor(reward * self._conf.reward_scale, dtype=torch.float32)

            # Get the next state
            s_next = torch.tensor(vec_state, dtype=torch.float32)
            # --------------- Environment's job to provide info ---------------

            # Add it to the states
            while s_next.dim() < self._states.dim():
                s_next = s_next.unsqueeze(0)
            self._states = torch.cat([self._states, s_next.to(self._device)], dim=1)

            # Add it to the actions
            while action.dim() < self._actions.dim():
                action = action.unsqueeze(0)
            self._actions = torch.cat([self._actions, action.to(self._device)], dim=1)

            # Adjust the maximum length of the states and actions
            self._states = self._states[:, -self._conf.max_len:, :]
            self._actions = self._actions[:, -self._conf.max_len:, :]

            # Arrange the next observation as the model expects
            next_policy_obs, next_value_fn_obs = self.prettify_observation(self._states, self._actions)

            return curr_policy_obs, curr_value_fn_obs, next_policy_obs, next_value_fn_obs, r, False
        
    def prettify_observation(self, states: torch.Tensor, actions: torch.Tensor):
        """
        Arrange the next observation as the model expects.
        """
        # Clone the tensors to avoid in-place operations
        states = states.clone()
        actions = actions.clone()

        sequential_models = ["Transformer", "TransformerEncoder", "DiscreteStateTransformerEncoder"]

        # See if the policy is a sequential model
        if self._conf.policy_arch in sequential_models:
            policy_obs = states
        else:
            policy_obs = states.view(-1)

        # Check if we have a transformer policy but not a transformer value function
        if self._conf.policy_arch == "Transformer" and self._conf.value_fn_arch != "Transformer":
            value_fn_obs = torch.cat([states, actions], dim=-1)
        else:
            value_fn_obs = states

        # See if the value function is a sequential model
        if self._conf.value_fn_arch not in sequential_models:
            value_fn_obs = value_fn_obs.view(-1)

        return policy_obs, value_fn_obs
        
    def test(self, n_steps: int=10000):
        """
        Test the environment.
        """
        total_rewards = []

        self._policy.eval()
        for i in range(int(n_steps)):
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                observation, _ = self.prettify_observation(self._states, self._actions)
                actions_as_tgt = self._actions.clone()
                loc, scale, action, log_prob = self._policy(observation) if self._conf.policy_arch != "Transformer" else self._policy(observation, actions_as_tgt)
                curr_policy_obs, curr_value_fn_obs, next_policy_obs, next_value_fn_obs, reward, done = self.move_once(action)

            if done:
                n_steps = i + 1
                total_rewards.append(reward.detach().item())
                break

            total_rewards.append(reward.detach().item())

        return sum(total_rewards)/n_steps
        
    def normalize_state(self, state: dict) -> list:
        """
        Normalize the state dictionary to a list.
        """
        # Conversion dictionary: each has two elements, the first is the gain and the second is the offset
        conversion_dict = {
            "a": (1/RT, -1), "e": (1, 0), "i": (1/180, 0), "raan": (1/360, 0), "aop": (1/360, 0), "ta": (1/360, 0), # orbital elements
            "az": (1/360, 0), "el": (1/180, 0.5), # azimuth and elevation
            "pitch": (1/180, 0.5), "roll": (1/360, 0.5), # attitude
            "detic_lat": (1/180, 0.5), "detic_lon": (1/360, 0.5), "detic_alt": (1/RT, 0), # nadir position
            "lat": (1/180, 0.5), "lon": (1/360, 0.5), "priority": (1/10, 0) # targets clues
        }

        vec_state = []
        for key, value in state.items():
            if key.startswith("lat_") or key.startswith("lon_") or key.startswith("priority_"):
                key = key.split("_")[0]
            vec_state.append(value * conversion_dict[key][0] + conversion_dict[key][1])

        # Check they are all between 0 and 1
        assert all([0 <= x <= 1 for x in vec_state]), f"State elements ∈ [0, 1] constraint violated: {vec_state}"

        return vec_state