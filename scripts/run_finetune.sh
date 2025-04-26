#!/usr/bin/env bash
if [ "$#" -ne 3 ]; then
  echo "Usage: run_finetune.sh <common.yaml> <td3_finetune.yaml> <es_checkpoint.pt>"
  exit 1
fi

COMMON_CFG=$1
TD3_CFG=$2
ES_CKPT=$3

python -m src.es_drl.main_finetune "$COMMON_CFG" "$TD3_CFG" --pretrained "$ES_CKPT"
