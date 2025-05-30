# ES-DRL: Evolution Strategies with Deep Reinforcement Learning

This project implements a combination of Evolution Strategies (ES) and Deep Reinforcement Learning (DRL) algorithms for training agents in MuJoCo environments. It provides multiple training strategies including basic ES, PPO, and pretraining (ES followed by PPO).

## Features

- Multiple training strategies:
  - Basic Evolution Strategy (ES)
  - Proximal Policy Optimization (PPO)
  - Pretraining (ES followed by PPO)
- Support for various MuJoCo environments
- Wandb integration for experiment tracking
- Video recording of agent performance
- Comprehensive logging and metrics tracking
- Early stopping based on reward thresholds
- GPU acceleration support
- Headless MuJoCo rendering support

## Project Structure

```
src/es_drl/
├── es/
│   ├── base.py            # Base ES class implementing core evolution strategy functionality
│   ├── basic_es.py        # Basic Evolution Strategy implementation with population-based optimization
│   ├── brax_training_utils.py  # Training utilities for ES in Brax environments
│   ├── ppo.py             # PPO implementation with clipped objective and GAE
│   ├── ppo_training_utils.py   # Training utilities for PPO including advantage computation
│   └── pretraining.py     # Combined ES+PPO pretraining with parameter transfer
├── utils/
│   ├── callbacks.py       # Training callbacks for logging and early stopping
│   └── logger.py          # Logging utilities for metrics and experiment tracking
└── main_es.py             # Main script for ES training with configuration handling
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training with ES

```bash
python src/es_drl/main_es.py --config path/to/config.yaml --seed 42 --env_id hopper
```

## Configuration

The project uses YAML configuration files for different components:

- Common configuration (`common.yaml`): Shared settings like environment ID, seeds, video settings
- ES configuration (`es_config.yaml`): ES-specific hyperparameters

### Example Configuration Structure

```yaml
# common.yaml
env_id: "hopper"
num_envs: 1
seed: 42
video:
  folder: "videos"
  freq: 10000
  length: 1000
total_timesteps: 1000000

# es_config.yaml
es_name: "basic_es"  # or "ppo", "pretraining"
sigma: 0.1
population_size: 128
learning_rate: 1e-3
num_timesteps: 1000000
hidden_sizes: [400, 300]
```

## Supported Environments

The project supports various MuJoCo environments:
- Ant
- HalfCheetah
- Hopper
- Humanoid
- HumanoidStandup
- Reacher
- Walker2d
- Pusher

## Training Process

### Evolution Strategies (ES)

The ES implementation follows a population-based optimization approach:

1. **Parameter Perturbation**:
   - Each iteration samples perturbations ε ~ N(0, I)
   - Parameters are perturbed: θ_i = θ + σ * ε_i
   - σ is the noise standard deviation

2. **Evaluation**:
   - Each perturbed policy is evaluated in the environment
   - Returns are collected and normalized

3. **Update**:
   - Gradient estimate: g = (1/nσ) * Σ(F_i * ε_i)
   - Parameter update: θ_new = θ + α * g
   - α is the learning rate

### PPO Training

The PPO implementation includes:
- Parallel environment execution
- Advantage normalization
- GAE (Generalized Advantage Estimation)
- Clipped objective function
- Value function loss
- Entropy bonus

### Pretraining Strategy

The pretraining approach combines ES and PPO:
1. Initial ES training for exploration
2. Parameter transfer to PPO actor network
3. Independent critic network initialization
4. PPO fine-tuning

## MuJoCo Environment Details

### Environment Configurations

1. **Walker2d**:
   - Timesteps: 7,864,320
   - Reward scaling: 5
   - Discount factor: 0.997
   - Learning rate: 6e-4
   - Batch size: 128
   - Gradient updates per step: 32

2. **HalfCheetah**:
   - Timesteps: 6,553,600
   - Reward scaling: 30
   - Batch size: 512
   - Gradient updates per step: 64

3. **Hopper**:
   - Timesteps: 6,553,600
   - Reward scaling: 30
   - Batch size: 512
   - Gradient updates per step: 64

### Model Architecture

- Fully connected neural network
- 4 hidden layers with 32 units each
- Non-linear activation functions
- Observation normalization
- Action repeat: 1 (fine-grained control)

## Monitoring and Logging

- Wandb integration for experiment tracking
- CSV logging of episode rewards
- Video recording of agent performance
- Early stopping based on reward thresholds
- Training metrics visualization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
