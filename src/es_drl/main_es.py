#!/usr/bin/env python
import sys
import yaml

from src.es_drl.es.basic_es import BasicES

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: main_es.py <common_config.yaml> <es_config.yaml>")
        sys.exit(1)

    common_cfg = load_yaml(sys.argv[1])
    es_cfg     = load_yaml(sys.argv[2])

    if es_cfg["es_name"] == "basic_es":
        es = BasicES(common_cfg, es_cfg)
    else:
        raise NotImplementedError(f"ES '{es_cfg['es_name']}' not implemented yet")

    ckpt = es.run()
    print(f"[ES] Training completed. Checkpoint saved at: {ckpt}")
