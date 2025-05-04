# src/es_drl/es/base.py
import os
from abc import ABC, abstractmethod


class EvolutionStrategy(ABC):
    def __init__(self, common_cfg: dict, es_cfg: dict):
        """
        common_cfg: loaded from configs/common.yaml
        es_cfg:   loaded from configs/es/<algo>.yaml
        """
        self.common_cfg = common_cfg
        self.es_cfg = es_cfg

        # Environment and seeds
        self.env_id = common_cfg["env_id"]
        self.seed = common_cfg["seed"]

        # Create directories for models, logs and videos
        self.es_name = es_cfg["es_name"]
        self.model_dir = os.path.join("models", "es", self.es_name)
        self.log_dir = os.path.join("logs", "es", self.es_name)
        self.video_dir = common_cfg["video"]["folder_es"]
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

    @abstractmethod
    def run(self) -> str:
        """
        Execute the ES training loop.
        Returns the path to the final checkpoint.
        """
        pass
