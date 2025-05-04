# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evolution strategy training.

See: https://arxiv.org/pdf/1703.03864.pdf
"""

from datetime import datetime
import os

import matplotlib.pyplot as plt
import jax
import imageio
from brax import envs
from brax.io import model, image

from src.es_drl.es import brax_training_utils
from src.es_drl.es.base import EvolutionStrategy
from src.es_drl.utils.logger import Logger

class BasicES(EvolutionStrategy):
    def __init__(self, common_cfg, es_cfg):
        super().__init__(common_cfg, es_cfg)

        self.hidden_sizes   = es_cfg.get("hidden_sizes", [400, 300])

        # ES hyperparameters
        self.sigma           = es_cfg["sigma"]
        self.population_size = es_cfg["population_size"]
        self.lr              = es_cfg["learning_rate"]
        self.num_timesteps = es_cfg["num_timesteps"]

        # Setup logger and video recording
        self.logger = Logger(self.log_dir)
        self.video_folder = common_cfg["video"]["folder_es"]
        self.video_freq   = common_cfg["video"]["freq_es"]
        self.video_length = common_cfg["video"]["length"]
        os.makedirs(self.video_folder, exist_ok=True)

        # Verbose flag
        self.verbose = es_cfg.get("verbose", False)

    def _save_video(self):
        # create an env with auto-reset
        env = envs.create(env_name=self.env_id)

        jit_env_reset = jax.jit(env.reset)
        jit_env_step = jax.jit(env.step)
        jit_inference_fn = jax.jit(self.inference_fn)

        rollout = []
        rng = jax.random.PRNGKey(seed=1)
        state = jit_env_reset(rng=rng)
        for _ in range(1000):
            rollout.append(state.pipeline_state)
            act_rng, rng = jax.random.split(rng)
            act, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_env_step(state, act)

        frames = image.render_array(env.sys, jax.device_get(rollout), height=480, width=640)
        fps = int(1.0 / env.dt)
        with imageio.get_writer(f"results/es/{self.env_id}/{self.es_name}_seed{self.seed}.mp4", fps=fps) as w:
            for frame in frames:
                w.append_data(frame)

    def run(self) -> str:
        xdata, ydata = [], []
        times = [datetime.now()]

        def progress(num_steps, metrics):
            times.append(datetime.now())
            xdata.append(num_steps)
            ydata.append(metrics['eval/episode_reward'])
            plt.xlim([0, self.num_timesteps])
            plt.ylim([min_y, max_y])
            plt.xlabel('# environment steps')
            plt.ylabel('reward per episode')
            plt.plot(xdata, ydata)
            print(f"Reward: {metrics['eval/episode_reward']}")
            plt.savefig(f"results/es/{self.env_id}/{self.es_name}_seed{self.seed}_{num_steps}.png")

        max_y = {'ant': 8000, 'halfcheetah': 8000, 'hopper': 2500, 'humanoid': 13000, 'humanoidstandup': 75_000, 'reacher': 5, 'walker2d': 5000, 'pusher': 0}[self.env_id]
        min_y = {'reacher': -100, 'pusher': -150}.get(self.env_id, 0)

        make_inference_fn, self.params, _ = brax_training_utils.train(
            environment=envs.get_environment(self.env_id),
            wrap_env=True,
            num_timesteps=self.num_timesteps,
            episode_length=1000,
            action_repeat=1,
            l2coeff=0,
            population_size=self.population_size,
            learning_rate=self.lr,
            fitness_shaping=brax_training_utils.FitnessShaping.WIERSTRA,
            num_eval_envs=128,
            perturbation_std=self.sigma,
            seed=self.seed,
            normalize_observations=True,
            num_evals=20,
            center_fitness=True,
            deterministic_eval=False,
            progress_fn=progress,
        )

        print(f'time to jit: {times[1] - times[0]}')
        print(f'time to train: {times[-1] - times[1]}')
        self.inference_fn = make_inference_fn(self.params)

        model.save_params(os.path.join(self.model_dir, f"{self.env_id}/{self.es_name}_seed{self.seed}.pt"), self.params)
        self._save_video()
