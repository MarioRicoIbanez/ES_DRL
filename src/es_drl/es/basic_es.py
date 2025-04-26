# src/es_drl/es/basic_es.py
import os
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from src.es_drl.es.base import EvolutionStrategy
from src.es_drl.utils.logger import Logger

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

class BasicES(EvolutionStrategy):
    def __init__(self, common_cfg, es_cfg):
        super().__init__(common_cfg, es_cfg)

        # Build policy network
        env = gym.make(self.env_id)
        self.action_high = env.action_space.high
        obs_dim    = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        env.close()

        self.hidden_sizes   = es_cfg.get("hidden_sizes", [400, 300])
        self.policy         = MLP(obs_dim, action_dim, self.hidden_sizes)

        # ES hyperparameters
        self.sigma           = es_cfg["sigma"]
        self.population_size = es_cfg["population_size"]
        self.elite_frac      = es_cfg.get("elite_frac", 0.5)
        self.lr              = es_cfg["learning_rate"]
        self.num_generations = es_cfg["num_generations"]

        # Setup logger
        self.logger = Logger(self.log_dir)

        # Seeding
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def _get_param_vector(self):
        return torch.nn.utils.parameters_to_vector(
            self.policy.parameters()
        ).detach().numpy()

    def _set_param_vector(self, vec: np.ndarray):
        tensor = torch.tensor(vec, dtype=torch.float32)
        torch.nn.utils.vector_to_parameters(tensor, self.policy.parameters())

    def _evaluate_candidate(self, params: np.ndarray) -> float:
        self._set_param_vector(params)
        env = gym.make(self.env_id)
        obs, _ = env.reset(seed=self.seed)
        done = False
        total_reward = 0.0
        while not done:
            with torch.no_grad():
                raw = self.policy(torch.tensor(obs, dtype=torch.float32)).numpy()
                action = np.tanh(raw) * self.action_high
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        env.close()
        return total_reward

    def run(self) -> str:
        # Initialize mean parameter vector μ
        mu = self._get_param_vector()
        param_dim = mu.shape[0]
        num_elite = max(1, int(self.elite_frac * self.population_size))

        for gen in range(self.num_generations):
            # Sample noise and evaluate
            noise = np.random.randn(self.population_size, param_dim)
            rewards = np.zeros(self.population_size)
            for i in range(self.population_size):
                candidate = mu + self.sigma * noise[i]
                rewards[i] = self._evaluate_candidate(candidate)

            # Select elite perturbations
            elite_idx   = np.argsort(rewards)[-num_elite:]
            elite_noise = noise[elite_idx]
            elite_rewards = rewards[elite_idx]

            # Update μ via weighted average of elite
            mu += (self.lr / (num_elite * self.sigma)) * (elite_noise.T @ elite_rewards)

            # Log progress
            mean_elite = float(np.mean(elite_rewards))
            self.logger.log(gen, {"reward_mean_elite": mean_elite})

        # Save final policy
        self._set_param_vector(mu)
        ckpt_path = os.path.join(self.model_dir, f"{self.es_name}_seed{self.seed}.pt")
        torch.save(self.policy.state_dict(), ckpt_path)
        return ckpt_path
