"""
Script with LLM model definitions 
# References:
- https://github.com/rasbt/LLMs-from-scratch
- https://github.com/labmlai/annotated_deep_learning_paper_implementations
- https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py
"""

import torch
import torch.nn as nn

from llm_components_gpt_llama import *
from llm_configs_1 import *

class GPT2TransformerBlock(nn.Module):
    def __init__(self, cfg):
        
        super().__init__()
        din, dim, seq_len = cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"]
        dropout, num_heads = cfg["drop_rate"], cfg["n_heads"]

        self.att =  MHA2(din, dim, seq_len, dropout, num_heads)
        self.ff = FFN(dim)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.drop_shortcut = nn.Dropout(dropout)

    def forward(self, x):

        # Skip connection for attention
        shortcut = x   # (bs, seq_len, dim)
        x = self.norm1(x)
        x = self.att(x)  
        x = self.drop_shortcut(x)
        x = x + shortcut 
        # Skip connection for ffn
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut 
        return x
    
class GPT2Model(nn.Module):
    def __init__(self, cfg):

        super().__init__()
        din, dim, seq_len = cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"]
        dropout, vocab_size, num_layers = cfg["drop_rate"], cfg["vocab_size"], cfg["n_layers"]

        self.tok_emb = nn.Embedding(vocab_size, din)
        self.pos_emb = nn.Embedding(seq_len, din)
        self.drop_emb = nn.Dropout(dropout)
        self.layers = nn.Sequential(*[GPT2TransformerBlock(cfg) for _ in range(num_layers)])
        self.final_norm = LayerNorm(dim)
        self.out_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_tokens):

        batch_size, seq_len = input_tokens.shape
        tok_embeds = self.tok_emb(input_tokens)
        positions = torch.arange(seq_len, device=input_tokens.device)
        pos_embeds = self.pos_emb(positions)
        x = tok_embeds + pos_embeds  # (bs, seq_len, dim)
        x = self.drop_emb(x)
        x = self.layers(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
# Example model instantiate
# cfg = GPT_CONFIG_124M
# model = GPT2Model(GPT_CONFIG_124M)
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params:,}")
# print(f"float32 : {model_memory_size(model, input_dtype=torch.float32):.2f} GB")
# print(f"bfloat16: {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")

class Llama3TransformerBlock(nn.Module):
    def __init__(self, cfg):

        super().__init__()
        din, dim, seq_len = cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"]
        dropout, num_heads = 0.0, cfg["n_heads"]
        rope_base, rope_config, dtype = cfg["rope_base"], cfg["rope_freq"], cfg["dtype"]
        ffn_hidden_dim = cfg["hidden_dim"]
        self.att =  GQARoPE(din, dim, seq_len, dropout, num_heads, num_kv_groups, bias=False, dtype=dtype, rope_base=rope_base, rope_config=rope_config)        
        self.ff = FFNSwiGLU(dim, ffn_hidden_dim)
        self.norm1 = RMSNorm(dim, eps=1e-5)
        self.norm2 = RMSNorm(dim, eps=1e-5)

    def forward(self, x):

        # Skip connection for attn
        shortcut = x   # (bs, seq_len, dim)
        x = self.norm1(x)
        x = self.att(x.to(torch.bfloat16))   # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut         
        # Skip connection for ffn
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x.to(torch.bfloat16))
        x = x + shortcut        
        return x
    
class Llama3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        din, dim, seq_len = cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"]
        dropout, vocab_size, num_layers = 0.0, cfg["vocab_size"], cfg["n_layers"]
        dtype = cfg["dtype"]

        self.tok_emb = nn.Embedding(vocab_size, dim, dtype=dtype)
        self.layers = nn.Sequential(*[Llama3TransformerBlock(cfg) for _ in range(num_layers)])
        self.final_norm = RMSNorm(dim, eps=1e-5)
        self.out_head = nn.Linear(dim, vocab_size, bias=False, dtype=dtype)

    def forward(self, input_tokens):

        tok_embeds = self.tok_emb(input_tokens)
        x = tok_embeds  # (bs, seq_len, dim)
        x = self.layers(x)
        x = self.final_norm(x)
        logits = self.out_head(x.to(torch.bfloat16))
        
        return logits

# # Change context length
# LLAMA32_CONFIG = LLAMA32_CONFIG_1B
# old_context_length = LLAMA32_CONFIG["context_length"]
# LLAMA32_CONFIG["context_length"] = 8192
# def rescale_theta(theta_old, context_length_old, context_length_new):
#     scaling_factor = context_length_new / context_length_old
#     theta_new = theta_old * scaling_factor
#     return theta_new
# print("Old RoPE theta:", LLAMA32_CONFIG["rope_base"])
# LLAMA32_CONFIG["rope_base"] = rescale_theta(
#     LLAMA32_CONFIG["rope_base"],
#     old_context_length,
#     LLAMA32_CONFIG["context_length"]
# )
# print("New RoPE theta:", LLAMA32_CONFIG["rope_base"])
# LLAMA32_CONFIG_1B = LLAMA32_CONFIG

# # Example model instantiate
# cfg = LLAMA32_CONFIG_1B
# model = Llama3Model(LLAMA32_CONFIG_1B)
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params:,}")
# print(f"float32 : {model_memory_size(model, input_dtype=torch.float32):.2f} GB")
# print(f"bfloat16: {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")
# # Account for weight tying
# total_params_normalized = total_params - model.tok_emb.weight.numel()
# print(f"\nTotal number of unique parameters: {total_params_normalized:,}")