# Env in short
```
python -m venv venv3a
source venv3a/bin/activate 
pip install -r requirements_updated.txt 
```

# Env instructions in detail
```
python -m venv venv3a
source venv3a/bin/activate 
pip install wheel
pip install -e .
pip install grouped_gemma
pip install flash-attn --no-build-isolation
pip install tensorboard
```

# Instructions that are common to Lora, full finetuning 
```
# config yaml
max_image_size: 980 #490
model_revision: 4844f0b5ff678e768236889df5accbe4967ec845
report_to: tensorboard #wandb
```

# Lora finetuning & inference / evaluation
```
CUDA_VISIBLE_DEVICES=6 python aria/train.py --config examples/nextqa/config_lora.yaml --output_dir outputs_lora/ &> logs/train_log_lora

CUDA_VISIBLE_DEVICES=6 python aria/inference.py \
    --base_model_path rhymes-ai/Aria \
    --tokenizer_path rhymes-ai/Aria \
    --image_path "./NExTVideo/1164/3238737531.mp4" \
    --prompt "Your prompt here" \
    --max_image_size 980 \
    --peft_model_path outputs_lora/checkpoint-1

CUDA_VISIBLE_DEVICES=6 python examples/nextqa/evaluation.py \
    --base_model_path rhymes-ai/Aria \
    --tokenizer_path  rhymes-ai/Aria \
    --save_root outputs_eval \
    --image_size 980 \
    --peft_model_path outputs_lora/checkpoint-1 # OPTIONAL
```

# Full finetuning & inference/evaluation
```
# zero yaml:
num_processes: 2 #8

CUDA_VISIBLE_DEVICES=6,7 accelerate launch --config_file recipes/accelerate_configs/zero3_offload.yaml aria/train.py --config examples/nextqa/config_full.yaml --output_dir outputs_full/  &> logs/train_log_full

CUDA_VISIBLE_DEVICES=6 python examples/nextqa/evaluation.py \
    --base_model_path [YOUR_ARIA_PATH] \
    --tokenizer_path [YOUR_ARIA_TOKENIZER_PATH] \
    --save_root [YOUR_SAVE_PATH] \
    --image_size [490] \
    --peft_model_path [YOUR_LORA_PATH] # OPTIONAL
```