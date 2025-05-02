# src/es_drl/es/basic_es.py
import os
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from joblib import Parallel, delayed

from src.es_drl.es.base import EvolutionStrategy
from src.es_drl.utils.logger import Logger


os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

"""
Implementation based on the following article:

https://lilianweng.github.io/posts/2019-09-05-evolution-strategies/#openai-es-for-rl
"""

class MLP(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class NSES(EvolutionStrategy):
    def __getstate__(self):
        """
        Exclude the logger (and file handles) from the pickled state
        so that Joblib workers don't try to serialize open files.
        """
        state = self.__dict__.copy()
        # Remove the logger and any file handle
        state.pop('logger', None)
        return state

    def __setstate__(self, state):
        """
        After unpickling in a worker process, restore attributes and
        recreate the logger so calls to self.logger.log(...) still work
        in the main process (but worker won't use it).
        """
        self.__dict__.update(state)
        from src.es_drl.utils.logger import Logger
        # Recreate logger pointing to the same log_dir
        self.logger = Logger(self.log_dir)

    def _set_up_environment(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        env = gym.make(self.env_id)
        action_high = env.action_space.high
        obs_dim    = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        env.close()
        return action_high, obs_dim, action_dim

    def __init__(self, common_cfg, es_cfg):
        super().__init__(common_cfg, es_cfg)

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build policy network and move to device
        (
            self.action_high, 
            obs_dim, 
            action_dim
        ) = self._set_up_environment()

        self.hidden_sizes   = es_cfg.get("hidden_sizes", [400, 300])
        self.policy         = MLP(obs_dim, action_dim, self.hidden_sizes).to(self.device)

        # ES hyperparameters
        self.sigma           = es_cfg["sigma"]
        self.population_size = es_cfg["population_size"]
        self.lr              = es_cfg["learning_rate"]
        self.num_generations = es_cfg["num_generations"]
        self.num_neighbors   = es_cfg["num_neighbors"]

        self.final_positions = set()

        # Setup logger and video recording
        self.logger = Logger(self.log_dir)
        self.video_folder = common_cfg["video"]["folder_es"]
        self.video_freq   = common_cfg["video"]["freq_es"]
        self.video_length = common_cfg["video"]["length"]
        os.makedirs(self.video_folder, exist_ok=True)

        # Verbose flag
        self.verbose = es_cfg.get("verbose", False)

        # Seeding
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _get_param_vector(self):
        return torch.nn.utils.parameters_to_vector(
            self.policy.parameters()
        )

    def _set_param_vector(self, vec: np.ndarray):
        torch.nn.utils.vector_to_parameters(vec, self.policy.parameters())
    
    def _compute_final_position(self, params: np.ndarray) -> float:
        self.policy.to(self.device)
        self._set_param_vector(params)

        env = gym.make(self.env_id)
        obs, _ = env.reset(seed=self.seed)
        done = False
        total_reward = 0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                raw = self.policy(obs_tensor)
                action = torch.tanh(raw).cpu().numpy() * self.action_high
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        env.close()
        return obs, total_reward

    def _compute_novelty_scores(self, candidates):
        output = (Parallel(n_jobs=-1)(
            delayed(self._compute_final_position)(cand) for cand in candidates
        ))

        final_positions = np.array([tuple[0] for tuple in output])
        rewards = np.array([tuple[1] for tuple in output])

        # print(final_positions.shape, rewards.shape)

        def dist(x, y):
            return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

        # Build distance matrix
        # N x N matrix
        distance_matrix = np.zeros((final_positions.shape[0], final_positions.shape[0]))
        for i in range(final_positions.shape[0]):
            for j in range(final_positions.shape[0]):
                distance_matrix[i, j] = dist(final_positions[i], final_positions[j])
        
        # Choose top k neighbors and compute mean distance
        novelty_scores = np.sort(distance_matrix, axis=0)[:self.num_neighbors, :].mean(axis=0)
        return novelty_scores, rewards

    def run(self) -> str:
        # Initialize mean parameter vector μ on device
        mu = self._get_param_vector()
        param_dim = mu.numel()
        agents = torch.randn(self.population_size, param_dim, device=self.device)

        for gen in range(self.num_generations):
            print(f'===GENERATION {gen}===')
            novelty_scores, rewards = self._compute_novelty_scores(agents)
            score_sum = np.sum(novelty_scores)
            probabilities = novelty_scores / score_sum

            # print(novelty_scores.shape, probabilities.shape, self.population_size)

            agent_index = np.random.choice(list(range(self.population_size)), p=probabilities)
            mu = agents[agent_index]

            noise = torch.randn(self.population_size, param_dim, device=self.device)
            candidates = (mu.unsqueeze(0) + self.sigma * noise)
            candidate_novelty_scores, candidate_rewards = self._compute_novelty_scores(candidates)

            mu = mu + (self.lr / self.sigma) * (noise.t() @ candidate_novelty_scores)
            agents[agent_index] = mu

            # Log progress
            mean_reward = float(np.mean(candidate_rewards))
            rewards[agent_index] = mean_reward
            self.logger.log(gen, {"reward_mean": mean_reward})
            if self.verbose:
                print(f"[ES] AGENT REWARDS = {rewards}", flush=True)


            # Video recording every `video_freq` generations
            if hasattr(self, "video_freq") and (gen % self.video_freq == 0):
                # print(f"[ES] Recording video at generation {gen}")
                # Create a fresh env wrapped with VecVideoRecorder
                from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
                from stable_baselines3.common.monitor import Monitor

                # Set up single-env for video
                record_env = DummyVecEnv([
                    lambda: Monitor(
                        gym.make(self.env_id, render_mode="rgb_array"),
                        filename=None
                    )
                ])
                record_env = VecVideoRecorder(
                    record_env,
                    video_folder=self.video_folder,
                    record_video_trigger=lambda x: x == 0,
                    video_length=self.video_length,
                    name_prefix=f"ns_es-gen{gen}"
                )

                # Rollout with current mu deterministically
                obs = record_env.reset() 
                for _ in range(self.video_length):
                    # compute action from mu
                    # get obs as numpy, convert to tensor
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                    raw = self.policy(obs_tensor).cpu().detach().numpy()
                    action = np.tanh(raw) * self.action_high
                    obs, _, dones, _ = record_env.step(action)
                    if dones[0]:
                        break
                record_env.close()

        # Save final policy
        best_mu = agents[np.argmax(rewards)]
        self._set_param_vector(best_mu.cpu().numpy())
        ckpt_path = os.path.join(self.model_dir, f"{self.es_name}_seed{self.seed}.pt")
        torch.save(self.policy.state_dict(), ckpt_path)
        return ckpt_path
