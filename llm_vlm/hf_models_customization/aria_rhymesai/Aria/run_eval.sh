CUDA_VISIBLE_DEVICES=6 python examples/nextqa/evaluation.py \
    --base_model_path rhymes-ai/Aria \
    --tokenizer_path  rhymes-ai/Aria \
    --save_root outputs_eval \
    --image_size 980 \
    --peft_model_path outputs_lora/checkpoint-1