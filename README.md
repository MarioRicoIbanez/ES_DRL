# ES_DRL

Currently implemented ES:

## Basic Evolution Strategy (µ+λ-ES)

This project implements a **basic (µ+λ) Evolution Strategy** for continuous control tasks.  
At each iteration, a Gaussian distribution centered at the current solution μ with fixed isotropic noise σ is used to sample λ offspring parameter vectors θ.  
These candidates are evaluated in the environment (e.g., MuJoCo simulators), and the next center μ is updated based on the best-performing solutions.

Mathematically, each offspring is sampled as:

$$
\theta_i = \mu^{(t)} + \sigma^{(t)} \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, I)
$$

After evaluation, μ is updated to either the best offspring (plus optionally the parent) or to the mean of the best candidates.  
This simple ES variant is lightweight, requires no gradient computations, and is effective for optimizing small policy networks in continuous environments.

The code follows a simple design and allows direct finetuning with RL methods like TD3 after initial training.

---

## How to Execute

First, set up your environment:

```bash
source scripts/env.sh
```

If everything is correct, you can run the code as follows:

- **Run basic ES:**

```bash
./scripts/run_es.sh configs/common.yaml configs/es/basic_es.yaml
```

- **Run TD3 naive:**

```bash
./scripts/run_finetune.sh   configs/common.yaml   configs/td3/td3_finetune.yaml   --no-pretrained
```

- **Run TD3 based on Basic ES weights:**

```bash
./scripts/run_finetune.sh configs/common.yaml configs/td3/td3_finetune.yaml models/es/basic_es/basic_es_seed42.pt
```