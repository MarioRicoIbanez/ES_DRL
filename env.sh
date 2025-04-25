# Crear e iniciar un nuevo entorno conda (ej. con Python 3.9 o 3.10)
conda create -n mujoco_env python=3.10 -y
conda activate mujoco_env

# [A] Instalar MuJoCo, Gymnasium y Stable-Baselines3 (por pip o conda-forge)
pip install mujoco gymnasium stable-baselines3

# (Opcional) Instalar PyTorch si usarás GPU para entrenamiento con SB3
# por ejemplo, usando conda (ajusta la versión CUDA según el cluster):
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

# [B] Instalar librerías de renderizado offscreen (EGL/Mesa) y FFmpeg sin apt:
conda install -c conda-forge glew mesalib ffmpeg -y
conda install -c anaconda mesa-libegl-cos6-x86_64 mesa-libgl-cos6-x86_64 -y

# [C] (Opcional) Instalar GLFW si se requiere abrir ventanas (no en modo headless EGL):
conda install -c menpo glfw3 -y

# [D] Configurar variables de entorno para usar EGL en MuJoCo (render sin pantalla):
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Asegurar que las bibliotecas de conda (Mesa EGL/GL) estén en el PATH de carga:
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
