# Create and start a new conda environment (e.g., with Python 3.9 or 3.10)
conda create -n mujoco_env python=3.10 -y
conda activate mujoco_env

# [A] Install MuJoCo, Gymnasium, and Stable-Baselines3 (via pip or conda-forge)
pip install mujoco gymnasium stable-baselines3

# (Optional) Install PyTorch if you will use GPU for training with SB3
# for example, using conda (adjust the CUDA version depending on the cluster):
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

# [B] Install offscreen rendering libraries (EGL/Mesa) and FFmpeg without using apt:
conda install -c conda-forge glew mesalib ffmpeg -y
conda install -c anaconda mesa-libegl-cos6-x86_64 mesa-libgl-cos6-x86_64 -y

# [C] (Optional) Install GLFW if you need to open windows (not in headless EGL mode):
conda install -c menpo glfw3 -y

# [D] Set environment variables to use EGL in MuJoCo (headless rendering):
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Make sure conda libraries (Mesa EGL/GL) are in the loading PATH:
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# [E] Install all Python dependencies from requirements.txt
pip install -r ../requirements.txt