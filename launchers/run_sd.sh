#!/bin/bash

# Exit the script if any command fails
set -e
#SD1.5
export MODEL_NAME="runwayml/stable-diffusion-v1-5"

#SDXL
# export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
# export VAE="madebyollin/sdxl-vae-fp16-fix"

export DATASET_NAME="yuvalkirstain/pickapic_v2"
export CACHE_DIR="/raid/workspace/cvml_user/collaboration/dpo/pickv2"

BATCH_SIZE=4
GAC=128
REF_UPDATE_STEP=$(expr $GAC \* 32) # e.g. {16,32,64}
MONITOR_THRESHOLD=0.005 #0.002 for SDXL

#Recommend 4 or 8 gpus
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port=21205 --mixed_precision="fp16" train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=$BATCH_SIZE \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=$GAC \
  --max_train_steps=1000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=200 \
  --learning_rate=1e-8 --scale_lr \
  --cache_dir=$CACHE_DIR \
  --checkpointing_steps 500 \
  --beta_dpo 5000 \
  --output_dir="ckpt/dpo_ours" \
  --reference_update_method='hardstep'  \
  --ref_update_steps $REF_UPDATE_STEP \
  --monitor_threshold $MONITOR_THRESHOLD \
  --timestep_gamma 0.9 \
  --reward_scale_scheduling \
  --alpha 1
  