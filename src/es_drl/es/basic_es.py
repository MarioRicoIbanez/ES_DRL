# src/es_drl/es/basic_es.py
import os
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from joblib import Parallel, delayed

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

    def __init__(self, common_cfg, es_cfg):
        super().__init__(common_cfg, es_cfg)

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build policy network and move to device
        env = gym.make(self.env_id)
        self.action_high = env.action_space.high
        obs_dim    = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        env.close()

        self.hidden_sizes   = es_cfg.get("hidden_sizes", [400, 300])
        self.policy         = MLP(obs_dim, action_dim, self.hidden_sizes).to(self.device)

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
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _get_param_vector(self):
        return torch.nn.utils.parameters_to_vector(
            self.policy.parameters()
        ).detach().cpu().numpy()

    def _set_param_vector(self, vec: np.ndarray):
        tensor = torch.tensor(vec, dtype=torch.float32, device=self.device)
        torch.nn.utils.vector_to_parameters(tensor, self.policy.parameters())

    def _evaluate_candidate(self, params: np.ndarray) -> float:
        # Ensure policy on correct device
        self.policy.to(self.device)
        # Load candidate parameters
        self._set_param_vector(params)
        env = gym.make(self.env_id)
        obs, _ = env.reset(seed=self.seed)
        done = False
        total_reward = 0.0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                raw = self.policy(obs_tensor)
                action = torch.tanh(raw).cpu().numpy() * self.action_high
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        env.close()
        return total_reward

    def run(self) -> str:
        # Initialize mean parameter vector μ on device
        mu = torch.from_numpy(self._get_param_vector()).to(self.device)
        param_dim = mu.numel()
        num_elite = max(1, int(self.elite_frac * self.population_size))
        
        import time

        for gen in range(self.num_generations):

            # print(f"---Running GENERATION {gen}---")
            init_time = time.time()

            time1 = time.time()
            # Sample noise on device
            noise = torch.randn(self.population_size, param_dim, device=self.device)
            # Create candidate vectors on host for evaluation

            # Parallel evaluation (subprocess recreates logger)
            candidates = (mu.unsqueeze(0) + self.sigma * noise).cpu().numpy()
            rewards = Parallel(n_jobs=-1)(
                delayed(self._evaluate_candidate)(cand) for cand in candidates
            )

            rewards = np.array(rewards)
            # Ensure rewards are float32 to match model dtype
            rewards = rewards.astype(np.float32)
            # print(f"TIME EVALUATING FOR THIS GEN: {round((time.time() - time1), 4)}")

            # Select elite perturbations
            elite_idx     = np.argsort(rewards)[-num_elite:]
            elite_noise   = noise[elite_idx]         # tensor on device
            elite_rewards = torch.from_numpy(rewards[elite_idx]).to(self.device)

            # Update μ via weighted average of elite
            mu = mu + (self.lr / (num_elite * self.sigma)) * (elite_noise.t() @ elite_rewards)

            # Log progress
            mean_elite = float(torch.mean(elite_rewards))
            self.logger.log(gen, {"reward_mean_elite": mean_elite})
            # print(f"TIME: {round((time.time() - init_time), 4)}")

        # Save final policy
        self._set_param_vector(mu.cpu().numpy())
        ckpt_path = os.path.join(self.model_dir, f"{self.es_name}_seed{self.seed}.pt")
        torch.save(self.policy.state_dict(), ckpt_path)
        return ckpt_path
