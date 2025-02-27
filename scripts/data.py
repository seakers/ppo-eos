import torch
import numpy as np

from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor

RT = 6371.0 # Earth radius in km

class DataCollectorFromEarthGym():
    def __init__(
            self,
            client: any,
            conf: object,
            policy: any,
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

    def __iter__(self):
        """
        Returns the iterator object.
        """
        return self
    
    def __next__(self):
        """
        Collects data from the environment using the policy.
        """
        if self._current_step == 0:
            self.initialize_env()

        # Fields from main td
        _action = []
        _done = []
        _loc = []
        _observation = []
        _sample_log_prob = []
        _scale = []
        _step_count = []
        _terminated = []
        _truncated = []
        
        # Fields from collector td
        _collector_traj_id = []

        # Fields from next td
        _next_done = []
        _next_observation = []
        _next_reward = []
        _next_step_count = []
        _next_terminated = []
        _next_truncated = []

        if self._current_step < self._total_steps:
            for _ in range(self._batch_steps):
                with set_exploration_type(ExplorationType.RANDOM):
                    loc, scale, action, log_prob = self._policy(self._states.to(self._device))
                    next_observation, reward, done = self.move_once(action.reshape(-1))

                if done:
                    raise StopIteration

                terminated = truncated = done

                # Fields from main td
                _action.append(action)
                _done.append(False)
                _loc.append(loc)
                _observation.append(self._current_observation.astype(np.float32))
                _sample_log_prob.append(log_prob)
                _scale.append(scale)
                _step_count.append(self._step_count)
                _terminated.append(False)
                _truncated.append(False)

                # Fields from collector td
                _collector_traj_id.append(self._traj_id)

                # Fields from next td
                _next_done.append(terminated or truncated)
                _next_observation.append(next_observation.astype(np.float32))
                _next_reward.append(np.float32(reward))
                _next_step_count.append(self._step_count + 1)
                _next_terminated.append(terminated)
                _next_truncated.append(truncated)

                self._step_count += 1
                self._current_step += 1

            return TensorDict({
                "action": torch.stack(_action, dim=0).clone().detach().to(self._device),
                "done": torch.tensor(_done, device=self._device),
                "loc": torch.stack(_loc, dim=0).clone().detach().to(self._device),
                "observation": torch.as_tensor(np.array(_observation), device=self._device),  # if _observation is a list of numpy arrays
                "sample_log_prob": torch.stack(_sample_log_prob, dim=0).clone().detach().to(self._device),
                "scale": torch.stack(_scale, dim=0).clone().detach().to(self._device),
                "step_count": torch.tensor(_step_count, device=self._device),
                "terminated": torch.tensor(_terminated, device=self._device),
                "truncated": torch.tensor(_truncated, device=self._device),
                
                "collector": TensorDict({
                    "traj_ids": torch.tensor(_collector_traj_id, device=self._device)
                }, batch_size=self._batch_steps),

                "next": TensorDict({
                    "done": torch.tensor(_next_done, device=self._device),
                    "observation": torch.as_tensor(np.array(_next_observation), device=self._device),
                    "reward": torch.tensor(_next_reward, device=self._device),
                    "step_count": torch.tensor(_next_step_count, device=self._device),
                    "terminated": torch.tensor(_next_terminated, device=self._device),
                    "truncated": torch.tensor(_next_truncated, device=self._device),
                }, batch_size=self._batch_steps),
            }, batch_size=self._batch_steps)
        else:
            raise StopIteration
    
    def initialize_env(self):
        """
        Initialize the environment.
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
        self._states = torch.FloatTensor(vec_state)

        # Initialize the current observation
        self._current_observation = self._states.clone().view(-1).detach().numpy()
        self._current_observation = np.concatenate([self._current_observation, np.zeros(self._conf.max_len * self._conf.state_dim - len(self._current_observation))], axis=0)

    def move_once(self, action: torch.Tensor):
        """
        Do an environment step for the SAC algorithm.
        """
        with torch.no_grad():
            # --------------- Environment's job to provide info ---------------
            sending_data = {
                "agent_id": 0,
                "action": {
                    "d_pitch": action[0].item() * self._conf.a_conversions[0],
                    "d_roll": action[1].item() * self._conf.a_conversions[1]
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
            r = torch.FloatTensor([reward * self._conf.reward_scale])

            # Get the next state
            s_next = torch.FloatTensor(vec_state)
            # --------------- Environment's job to provide info ---------------

            # Update the current obnservation
            self._current_observation = self._states.clone().view(-1).detach().numpy()
            self._current_observation = np.concatenate([self._current_observation, np.zeros(self._conf.max_len * self._conf.state_dim - len(self._current_observation))], axis=0)

            # Add it to the states
            self._states = torch.cat([self._states, s_next.to(self._device)], dim=-1)

            # Adjust the maximum length of the states and actions
            self._states = self._states[-self._conf.max_len:]

            # Next observation
            next_observation = self._states.clone().view(-1).detach().numpy()
            next_observation = np.concatenate([self._current_observation, np.zeros(self._conf.max_len * self._conf.state_dim - len(self._current_observation))], axis=0)

            return next_observation, r, False
        
    def normalize_state(self, state: dict) -> list:
        """
        Normalize the state dictionary to a list.
        """
        # Conversion dictionary: each has two elements, the first is the gain and the second is the offset
        conversion_dict = {
            "a": (1/RT, -1), "e": (1, 0), "i": (1/180, 0), "raan": (1/360, 0), "aop": (1/360, 0), "ta": (1/360, 0), # orbital elements
            "az": (1/360, 0), "el": (1/180, 0.5), # azimuth and elevation
            "pitch": (1/180, 0.5), "roll": (1/360, 0.5), # attitude
            "detic_lat": (1/180, 0.5), "detic_lon": (1/360, 0), "detic_alt": (1/RT, 0), # nadir position
            "lat": (1/180, 0.5), "lon": (1/360, 0), "priority": (1/10, 0) # targets clues
        }

        vec_state = []
        for key, value in state.items():
            if key.startswith("lat_") or key.startswith("lon_") or key.startswith("priority_"):
                key = key.split("_")[0]
            vec_state.append(value * conversion_dict[key][0] + conversion_dict[key][1])

        return vec_state