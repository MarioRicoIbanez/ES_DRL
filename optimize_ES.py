import gymnasium as gym
import torch
import numpy as np
from torch import nn
from torch.func import functional_call, vmap

class MLP(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = gym.make("Humanoid-v5")
action_high = env.action_space.high
obs_dim    = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# 1) script or compile the *code* of your policy
template = MLP(obs_dim, action_dim, hidden_sizes=[64, 64]).double()
param_vector  = torch.nn.utils.parameters_to_vector(template.parameters()).detach()
template = torch.compile(template.eval(), mode="reduce-overhead").to(device).float()

param_shapes  = [p.shape for p in template.parameters()]
param_numels  = [p.numel() for p in template.parameters()]
cum_sizes     = torch.tensor([0] + list(torch.cumsum(torch.tensor(param_numels), 0)))

def unflatten_params(flat: torch.Tensor):
    """
    Given flat of shape (D,), returns a dict mapping each param name
    to a tensor of shape param_shapes[i].
    """
    sd = {}
    for i, (name, p) in enumerate(template.named_parameters()):
        start = cum_sizes[i].item()
        end   = cum_sizes[i+1].item()
        chunk = flat[start:end].view(param_shapes[i]).to(p.device, p.dtype)
        sd[name] = chunk
    return sd

def forward_from_vector(param_vectors: torch.Tensor, obs: torch.Tensor):
    state_dict = unflatten_params(param_vectors)
    return functional_call(template, state_dict, (obs,))

# 2) vmapped version: returns [P, action_dim]
batched_policy = vmap(forward_from_vector, in_dims=(0, 0))

def evaluate_candidate(candidates) -> float:
    print("EVALUATE CANDIDATE")
    obs, _ = envs.reset()
    done = np.zeros(num_envs, dtype=bool)
    total_reward = np.zeros(num_envs, dtype=float)
    
    while not done.all():
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)                    
        actions = batched_policy(candidates, obs_tensor).cpu().numpy() * action_high 
      
        obs, rewards, term, trunc, _ = envs.step(actions)
        still_running = ~done
        total_reward[still_running] += rewards[still_running]

        done |= (term | trunc)
    return total_reward

import time
# 3) one big vectorized env of size P
num_envs = 14
sigma = 0.01
learning_rate = 0.02
envs = gym.make_vec("Humanoid-v5", num_envs=num_envs, vectorization_mode="sync")
obs, _ = envs.reset()
param_vectors = param_vector.unsqueeze(0).repeat(num_envs, 1)
param_dim = param_vector.numel()


# 4) one-shot batched rollout step
for gen in range(10000):
    print(f"[ES] GENERATION {gen} START", flush=True)

    time1 = time.time()
    noise = torch.randn(num_envs, param_dim, device=device)
    candidates = (param_vectors + sigma * noise)

    rewards = evaluate_candidate(candidates)
    print(f"[ES] TIME EVALUATING FOR THIS GEN: {round((time.time() - time1), 4)}", flush=True)

    # Update μ via weighted average of elite
    param_vectors = param_vectors + (learning_rate / (num_envs * sigma)) * (noise.t() @ rewards)

    # Log progress
    mean_elite = float(np.mean(rewards))
    # self.logger.log(gen, {"reward_mean_elite": mean_elite})
    print(f"[ES] MEAN ELITE = {mean_elite:.3f}", flush=True)



