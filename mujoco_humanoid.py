import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

# ---- CONFIGURA AQUÍ ----
STEP_INTERVAL   = 20_000        # graba cuando (global_step % STEP_INTERVAL == 0)
VIDEO_LENGTH    = 1_000         # nº de pasos que se guardan en cada vídeo
VIDEO_DIR       = "videos_step" # carpeta de salida
# ------------------------

# 1️⃣  constructor que crea el entorno con renderizado off-screen
def make_env():
    return gym.make("Humanoid-v5", render_mode="rgb_array")

# 2️⃣  vectoriza (obligatorio para usar VecVideoRecorder)
vec_env = DummyVecEnv([make_env])

# 3️⃣  envuelto con grabación basada en pasos
vec_env = VecVideoRecorder(
    vec_env,
    video_folder=VIDEO_DIR,
    record_video_trigger=lambda step: step % STEP_INTERVAL == 0,
    video_length=VIDEO_LENGTH,
    name_prefix="humanoid_td3_step"
)

# 4️⃣  entrena el agente
model = TD3("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=100_000)   # ajusta según tus recursos
vec_env.close()                        # cierra, termina vídeos
