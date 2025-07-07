# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# copied from https://github.com/pytorch-labs/gpt-fast/blob/main/scripts/convert_hf_checkpoint.py
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file as load_safetensors_file


@dataclass
class ModelArgs:
    block_size: int = 16384
    vocab_size: int = 100352
    n_layer: int = 28
    n_head: int = 20
    dim: int = 2560
    intermediate_size: int = 1664
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    use_scaled_rope: bool = False
    num_experts: int = 64
    router_topk: int = 6
    num_shared_experts: int = 2
    image_token_index: int = 9

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        self.head_dim = self.dim // self.n_head


@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path(
        "checkpoints/meta-Transformer/Transformer-2-7b-chat-hf"
    ),
    model_name: Optional[str] = None,
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.name
    config = ModelArgs(
        vocab_size=100352,
        block_size=2048,
        n_layer=28,
        n_head=20,
        dim=2560,
        intermediate_size=1664,
    )

    # config = ModelArgs.from_name(model_name)
    print(f"Model config {config.__dict__}")

    # Load the json file containing weight mapping
    model_map_json_safetensors = checkpoint_dir / "model.safetensors.index.json"
    model_map_json_pytorch = checkpoint_dir / "pytorch_model.bin.index.json"
    model_map_json = None

    try:
        assert model_map_json_safetensors.is_file()
        model_map_json = model_map_json_safetensors
        print(f"Found safetensors index at {model_map_json_safetensors}")
    except AssertionError:
        print(f"{model_map_json_safetensors} not found")
    if model_map_json is None:
        try:
            assert model_map_json_pytorch.is_file()
            model_map_json = model_map_json_pytorch
            print(f"Found pytorch index at {model_map_json_pytorch}")
        except AssertionError:
            print(f"{model_map_json_pytorch} not found")

    if model_map_json is None:
        raise Exception("No model map found!")

    with open(model_map_json) as json_map:
        bin_index = json.load(json_map)

    weight_map = {
        "language_model.model.embed_tokens.weight": "llm.tok_embeddings.weight",
        "language_model.model.layers.{}.self_attn.q_proj.weight": "llm.layers.{}.attention.wq.weight",
        "language_model.model.layers.{}.self_attn.k_proj.weight": "llm.layers.{}.attention.wk.weight",
        "language_model.model.layers.{}.self_attn.v_proj.weight": "llm.layers.{}.attention.wv.weight",
        "language_model.model.layers.{}.self_attn.o_proj.weight": "llm.layers.{}.attention.wo.weight",
        "language_model.model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "language_model.model.layers.{}.mlp.router.weight": "llm.layers.{}.feed_forward.gate.weight",
        "language_model.model.layers.{}.mlp.experts.fc1.weight": "llm.layers.{}.feed_forward.cond_ffn.w1_w3",
        "language_model.model.layers.{}.mlp.experts.fc2.weight": "llm.layers.{}.feed_forward.cond_ffn.w2",
        "language_model.model.layers.{}.mlp.shared_experts.gate_proj.weight": "llm.layers.{}.feed_forward.shared_ffn.w1.weight",
        "language_model.model.layers.{}.mlp.shared_experts.up_proj.weight": "llm.layers.{}.feed_forward.shared_ffn.w3.weight",
        "language_model.model.layers.{}.mlp.shared_experts.down_proj.weight": "llm.layers.{}.feed_forward.shared_ffn.w2.weight",
        "language_model.model.layers.{}.input_layernorm.weight": "llm.layers.{}.attention_norm.weight",
        "language_model.model.layers.{}.post_attention_layernorm.weight": "llm.layers.{}.ffn_norm.weight",
        "language_model.model.norm.weight": "llm.norm.weight",
        "language_model.lm_head.weight": "llm.output.weight",
    }
    bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}

    def permute(w, n_head):
        dim = config.dim
        return (
            w.view(n_head, 2, config.head_dim // 2, dim)
            .transpose(1, 2)
            .reshape(config.head_dim * n_head, dim)
        )

    merged_result = {}
    for file in sorted(bin_files):
        if "safetensors" in str(file):
            state_dict = load_safetensors_file(str(file), device="cpu")
            merged_result.update(state_dict)
        else:
            state_dict = torch.load(
                str(file), map_location="cpu", mmap=True, weights_only=True
            )
            merged_result.update(state_dict)
    final_result = {}
    for key, value in merged_result.items():
        if "multi_modal_projector" in key or "vision_tower" in key:
            final_result[key] = value
            continue
        if "layers" in key:
            abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
            layer_num = re.search(r"\d+", key).group(0)
            new_key = weight_map[abstract_key]
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map[key]
        print(new_key)
        final_result[new_key] = value

    for key in tuple(final_result.keys()):
        if "wq" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            q = permute(q, config.n_head)
            k = permute(k, config.n_local_heads)
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v])
            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]
        if "cond_ffn.w1_w3" in key:
            w = torch.chunk(final_result[key], 2, dim=-1)
            final_result[key.replace("w1_w3", "w1")] = w[0].transpose(1, 2).contiguous()
            final_result[key.replace("w1_w3", "w3")] = w[1].transpose(1, 2).contiguous()
            del final_result[key]
        if "cond_ffn.w2" in key:
            final_result[key] = final_result[key].transpose(1, 2).contiguous()

    print(f"Saving checkpoint to {checkpoint_dir / 'model.pth'}")
    torch.save(final_result, checkpoint_dir / "model.pth")
    if "llama-3-" in model_name.lower() or "llama-3.1-" in model_name.lower():
        if "llama-3.1-405b" in model_name.lower():
            original_dir = checkpoint_dir / "original" / "mp16"
        else:
            original_dir = checkpoint_dir / "original"
        tokenizer_model = original_dir / "tokenizer.model"
        tokenizer_model_tiktoken = checkpoint_dir / "tokenizer.model"
        print(f"Copying {tokenizer_model} to {tokenizer_model_tiktoken}")
        shutil.copy(tokenizer_model, tokenizer_model_tiktoken)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert HuggingFace checkpoint.")
    parser.add_argument("--checkpoint_dir", type=Path)
    parser.add_argument("--model_name", type=str, default=None)

    args = parser.parse_args()
    convert_hf_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
    )
