#!/usr/bin/env bash
# Minimal MuJoCo environment setup with JAX and utilities

conda create --name mujoco_environment python=3.10 -y
conda activate mujoco_environment

pip install jax[cuda]
pip install imageio[ffmpeg] mujoco mujoco_mjx matplotlib wandb brax joblib pre-commit
