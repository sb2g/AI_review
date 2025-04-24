import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math

torch.manual_seed(2024)

# ----------
# Activations Normalization
# ----------

# 2 parameters: scale, shift
class LayerNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # unbiased False => No Bessel's correction => variance formula has divide by n (not n-1)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        ln = self.scale*norm + self.shift
        return ln
#lnorm = LayerNorm(dim=x.shape[-1], eps=1e-5)
#lnorm_pt = nn.LayerNorm(x.shape[-1], eps=1e-5)

# 1 parameter: scale (no mean shift)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        mean = x.pow(2).mean(dim=-1, keepdim=True)
        norm = x * torch.rsqrt(mean + self.eps)
        rn = self.scale*norm
        return rn
# rms_norm = RMSNorm(dim=x.shape[-1], eps=1e-5)
# rms_norm_pt = nn.RMSNorm(x.shape[-1], eps=1e-5)


# ----------
# Activations
# ----------

class Sigmoid(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return 1./(1 + torch.exp(-x))

class Tanh(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return torch.tanh(x)
    
class ReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.maximum(x, torch.tensor(0.0))
    
class LeakyReLU(nn.Module):
    def __init__(self, negative_slope=1e-2):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return torch.where(x > 0, x, self.negative_slope * x)

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # approximation of x.cdf(N(x))
        return 0.5*x*(
                    1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3)))
                    )

class SiLU(nn.Module):
    def __init__(self,):
        super().__init__()
        
    def forward(self, x):
        return x * torch.sigmoid(x)


# sigmoid, tanh, relu, lrelu, gelu, silu = Sigmoid(), Tanh(), ReLU(), LeakyReLU(), GELU(), SiLU()
# y_sigmoid, y_tanh, y_relu, y_lrelu, y_gelu, y_silu = sigmoid(x), tanh(x), relu(x), lrelu(x), gelu(x), silu(x)
# x_act_tuples = [("Sigmoid", sigmoid(x), nn.functional.sigmoid(x)), 
#                 ("Relu", relu(x),  nn.functional.relu(x)), 
#                 ("LeakyRelu", lrelu(x), nn.functional.leaky_relu(x)),
#                 ("Gelu", gelu(x), nn.functional.gelu(x)),
#                 ("Silu", silu(x), nn.functional.silu(x))]


# ----------
# Feedforward layer
# ----------
# Regular FFN with GELU actvn
class FFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = 4*dim
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.layers(x)
# ffn = FFN(dim=x.shape[-1])
# x_op = ffn(x)

# SwiGLU FFN with SiLU actvn
class FFNSwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dtype=None, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, dtype=dtype, bias=bias)
        self.fc2 = nn.Linear(dim, hidden_dim, dtype=dtype, bias=bias)
        self.fc3 = nn.Linear(hidden_dim, dim, dtype=dtype, bias=bias)
        self.silu = SiLU()

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)
# ffn_swiglu = FFNSwiGLU(dim=x.shape[-1], hidden_dim=x.shape[-1]*4)
# x_op = ffn_swiglu(x)

# ----------
# Positional embeddings
# ----------

# Absolute fixed positional encoding
class AbsoluteFixedPositionalEncoding(nn.Module):

    def __init__(self, seq_len, dim, theta_base=10000.0):

        super().__init__()
        pe = torch.zeros(seq_len, dim)
        positions = torch.arange(0, seq_len).unsqueeze(1).float() # seq_len
        two_i = torch.arange(0, dim, 2, dtype=torch.float) # dim/2
        theta = torch.exp(two_i * -(math.log(theta_base) / dim)) # dim/2
        angles = positions * theta # (seq_len, dim/2)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        pe = pe.unsqueeze(0)  # (1, seq_len, dim)
        self.register_buffer("pe", pe)

    def forward(self, x):
        bs, seq_len, dim = x.size()
        x = x + self.pe[:, :seq_len]
        return x
# theta_base = 10000.0
# seq_len = 100
# dim = 64
# pe_layer = AbsoluteFixedPositionalEncoding(seq_len, dim, theta_base)
# pe = pe_layer.pe[0] # (seq_len, dim)
# print(pe.shape)
# x = torch.randn(2, 3, dim) # bs, seq_len, dim
# x = pe_layer(x)
# print(x.shape)

# Absolute + Learnable positional embeddings
class AbsoluteLearnablePositionalEmbedding(nn.Module):

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pe = nn.Embedding(seq_len, dim)

    def forward(self, x):
        bs, seq_len, dim = x.size()
        x = x + self.pe(torch.arange(seq_len))
        return x
# seq_len = 100
# dim = 64
# pe_layer = AbsoluteLearnablePositionalEmbedding(seq_len, dim)
# pe = pe_layer.pe # (seq_len, dim)
# print(pe.weight.shape)
# x = torch.randn(2, 3, dim) # bs, seq_len, dim
# x = pe_layer(x)
# print(x.shape)

# Rotary positional embeddings (RoPE)
class RotaryPositionalEmbeddings(nn.Module):

    def __init__(self, seq_len, dim, theta_base=10000, rope_config=None):
        super().__init__()
        positions = torch.arange(seq_len).float()
        two_i = torch.arange(0, dim, 2, dtype=torch.float)
        theta = 1. / (theta_base ** (two_i / dim))
        angles = positions[:, None] * theta[None, :] # (seq_len, dim/2)
        # or angles = torch.outer(positions, theta)  
        # or torch.ger(positions, theta)  
        # or angles = torch.einsum('n,d->nd', positions, theta)  
        angles2 = torch.cat([angles, angles], dim=1) # (seq_len, dim)    
        cos = torch.cos(angles2) # (1, 1, seq_len, dim)
        sin = torch.sin(angles2) # (1, 1, seq_len, dim) 
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        half_dim = dim // 2
        seq_len = x.shape[2]
        x1 = x[:, :, :, :half_dim] # (bs, num_heads, seq_len, dim/2)
        x2 = x[:, :, :, half_dim:] # (bs, num_heads, seq_len, dim/2)
        neg_half_x = torch.cat([-x2, x1], dim=-1) # (bs, num_heads, seq_len, dim)
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)      
        x_rope = (x * cos) + (neg_half_x * sin) # (bs, num_heads, seq_len, dim)
        return x_rope
# theta_base = 10000.0
# seq_len = 100
# dim = 64
# pe_layer = RotaryPositionalEmbeddings(seq_len, dim, theta_base)
# print(pe_layer.cos.shape, pe_layer.sin.shape) # (1, 1, seq_len, dim)
# x = torch.randn(2, 1, 3, dim) # bs, num_heads, seq_len, dim
# x = pe_layer(x)
# print(x.shape)


# ----------
# Attention mechanisms 
# ----------

class MHA2(nn.Module):
    def __init__(self, din, dim, seq_len, dropout, num_heads, bias=False, dtype=None):
        super().__init__()

        assert dim % num_heads == 0, "Given dim should be multiple of num_heads"
        self.head_dim = dim // num_heads
        self.num_heads, self.din, self.dim = num_heads, din, dim
        
        self.wq = nn.Linear(din, dim, bias=bias, dtype=dtype)
        self.wk = nn.Linear(din, dim, bias=bias, dtype=dtype) 
        self.wv = nn.Linear(din, dim, bias=bias, dtype=dtype)  
        
        att_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        self.register_buffer('att_mask', att_mask)

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim, bias=bias, dtype=dtype) # bias can be True here, even if qkv bias can be False

    def forward(self, x):
        bs, seq_len, din = x.shape
        
        q, k, v = self.wq(x), self.wk(x), self.wv(x)  # (bs, seq_len, dim)

        # Reshape to (bs, seq_len, num_heads, head_dim)
        q = q.view(bs, seq_len, self.num_heads, self.head_dim)
        k = k.view(bs, seq_len, self.num_heads, self.head_dim) 
        v = v.view(bs, seq_len, self.num_heads, self.head_dim) 

        # Reshape to calculate attn in parallel for all heads
        q = q.transpose(1, 2) # (bs, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2) # (bs, num_heads, seq_len, head_dim)
        v = v.transpose(1, 2) # (bs, num_heads, seq_len, head_dim)
        
        # att matrix mult along seq_len, head_dim. 
        att = q @ k.transpose(2, 3) # (bs, num_heads, seq_len, seq_len)
        
        # causal attn + dropout 
        att_mask = self.att_mask.bool()[:seq_len, :seq_len] # Select mask for seq_len & convert to bool
        att.masked_fill_(att_mask, -torch.inf)      
        att = torch.softmax(att / k.shape[-1]**0.5, dim=-1)
        att = self.dropout(att)
        
        # Calc context & reshape from (bs, num_heads, seq_len, head_dim) & then to (bs, seq_len, num_heads, head_dim)
        ctx = (att @ v).transpose(1, 2)
        
        # Concatenate heads to get (bs, seq_len, dim) & make it contiguous in memory
        ctx = ctx.contiguous().view(bs, seq_len, self.dim)
        ctx = self.proj(ctx)

        return ctx
    
# x = torch.randn(3, 4)
# batch_x = torch.stack((x, x), dim=0) # Create a batch of input
# din = batch_x.shape[2] # Input dim
# seq_len = 10 # Max ctx length supported
# dim = 16 # dim of att layer embeddings
# num_heads = 8
# mha_layer = MHA2(din, dim, seq_len, 0.1, num_heads)
# ctx = mha_layer(batch_x) 
# total_params = sum(p.numel() for p in mha_layer.parameters())
# print(f"Number of parameters: {total_params:,}")
# print(batch_x.shape)
# print(ctx.shape, ctx)  # bs, seqlen, dim


class GQA(nn.Module):

    def __init__(self, din, dim, seq_len, dropout, num_heads, num_kv_groups, bias=False, dtype=None):
        super().__init__()

        assert dim % num_heads == 0, "Given dim should be multiple of num_heads"
        self.head_dim = dim // num_heads
        self.num_heads, self.din, self.dim = num_heads, din, dim

        assert num_heads % num_kv_groups == 0, "Given num_heads should be multiple of num_kv_groups"
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups  # Number of query heads per k, v

        self.wq = nn.Linear(din, dim, bias=bias, dtype=dtype)
        self.wk = nn.Linear(din, num_kv_groups * self.head_dim, bias=bias, dtype=dtype)
        self.wv = nn.Linear(din, num_kv_groups * self.head_dim, bias=bias, dtype=dtype)

        att_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        self.register_buffer('att_mask', att_mask)

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim, bias=bias, dtype=dtype)

    def forward(self, x):
        bs, seq_len, din = x.shape
        
        q    = self.wq(x)               # (bs, seq_len, dim=num_heads*head_dim)
        k, v = self.wk(x), self.wv(x)   # (bs, seq_len, num_kv_groups*head_dim)

        # Reshape to (bs, seq_len, num_heads, head_dim)
        q = q.view(bs, seq_len, self.num_heads, self.head_dim)
        k = k.view(bs, seq_len, self.num_kv_groups, self.head_dim) 
        v = v.view(bs, seq_len, self.num_kv_groups, self.head_dim) 

        # Reshape to calculate attn in parallel for all heads
        q = q.transpose(1, 2) # (bs, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2) # (bs, num_kv_groups, seq_len, head_dim)
        v = v.transpose(1, 2) # (bs, num_kv_groups, seq_len, head_dim)

        # Replicate k, v to match num_heads for q.
        # E.g. groupsize = 4. [k1, k2] -> [k1, k1, k1, k1, k2, k2, k2, k2]
        k = k.repeat_interleave(self.group_size, dim=1)  # (bs, num_heads, seq_len, head_dim)
        v = v.repeat_interleave(self.group_size, dim=1)  # (bs, num_heads, seq_len, head_dim)

        # att matrix mult along seq_len, head_dim. 
        att = q @ k.transpose(2, 3) # (bs, num_heads, seq_len, seq_len)
        
        # causal attn + dropout 
        att_mask = self.att_mask.bool()[:seq_len, :seq_len] # Select mask for seq_len & convert to bool
        att.masked_fill_(att_mask, -torch.inf)      
        att = torch.softmax(att / k.shape[-1]**0.5, dim=-1)
        att = self.dropout(att)
        
        # Calc context & reshape from (bs, num_heads, seq_len, head_dim) & then to (bs, seq_len, num_heads, head_dim)
        ctx = (att @ v).transpose(1, 2)
        
        # Concatenate heads to get (bs, seq_len, dim) & make it contiguous in memory
        #ctx = ctx.contiguous().view(bs, seq_len, self.dim)
        ctx = ctx.reshape(bs, seq_len, self.dim)
        ctx = self.proj(ctx)

        return ctx

# x = torch.randn(3, 4)
# batch_x = torch.stack((x, x), dim=0) # Create a batch of input
# din = batch_x.shape[2] # Input dim
# seq_len = 10 # Max ctx length supported
# dim = 16 # dim of att layer embeddings
# num_heads = 8
# num_kv_groups = 2 # i.e. 4 q heads share 1 k,v 
# gqa_layer = GQA(din, dim, seq_len, 0.1, num_heads, num_kv_groups)
# ctx = gqa_layer(batch_x) 
# total_params = sum(p.numel() for p in gqa_layer.parameters())
# print(f"Number of parameters: {total_params:,}")
# print(batch_x.shape)
# print(ctx.shape, ctx)  # bs, seqlen, dim

class GQARoPE(nn.Module):

    def __init__(self, din, dim, seq_len, dropout, num_heads, num_kv_groups, bias=False, 
                 dtype=None, rope_base=10000.0, rope_config=None):
        super().__init__()

        assert dim % num_heads == 0, "Given dim should be multiple of num_heads"
        self.head_dim = dim // num_heads
        self.num_heads, self.din, self.dim = num_heads, din, dim

        assert num_heads % num_kv_groups == 0, "Given num_heads should be multiple of num_kv_groups"
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups  # Number of query heads per k, v

        self.wq = nn.Linear(din, dim, bias=bias, dtype=dtype)
        self.wk = nn.Linear(din, num_kv_groups * self.head_dim, bias=bias, dtype=dtype)
        self.wv = nn.Linear(din, num_kv_groups * self.head_dim, bias=bias, dtype=dtype)

        self.rope_layer = RotaryPositionalEmbeddings(seq_len, self.head_dim, theta_base=rope_base, rope_config=rope_config)

        att_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        self.register_buffer('att_mask', att_mask)

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim, bias=bias, dtype=dtype)

    def forward(self, x):
        bs, seq_len, din = x.shape
        
        q    = self.wq(x)               # (bs, seq_len, dim=num_heads*head_dim)
        k, v = self.wk(x), self.wv(x)   # (bs, seq_len, num_kv_groups*head_dim)

        # Reshape to (bs, seq_len, num_heads, head_dim)
        q = q.view(bs, seq_len, self.num_heads, self.head_dim)
        k = k.view(bs, seq_len, self.num_kv_groups, self.head_dim) 
        v = v.view(bs, seq_len, self.num_kv_groups, self.head_dim) 

        # Reshape to calculate attn in parallel for all heads
        q = q.transpose(1, 2) # (bs, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2) # (bs, num_kv_groups, seq_len, head_dim)
        v = v.transpose(1, 2) # (bs, num_kv_groups, seq_len, head_dim)

        # Apply RoPE
        k = self.rope_layer(k)
        q = self.rope_layer(q)

        # Replicate k, v to match num_heads for q.
        # E.g. groupsize = 4. 
        # [k1, k2] -> [k1, k1, k1, k1, k2, k2, k2, k2]
        # [v1, v2] -> [v1, v1, v1, v1, v2, v2, v2, v2]
        k = k.repeat_interleave(self.group_size, dim=1)  # (bs, num_heads, seq_len, head_dim)
        v = v.repeat_interleave(self.group_size, dim=1)  # (bs, num_heads, seq_len, head_dim)

        # att matrix mult along seq_len, head_dim. 
        att = q @ k.transpose(2, 3) # (bs, num_heads, seq_len, seq_len)
        
        # causal attn + dropout 
        att_mask = self.att_mask.bool()[:seq_len, :seq_len] # Select mask for seq_len & convert to bool
        att.masked_fill_(att_mask, -torch.inf)      
        att = torch.softmax(att / k.shape[-1]**0.5, dim=-1)
        att = self.dropout(att)
        
        # Calc context & reshape from (bs, num_heads, seq_len, head_dim) & then to (bs, seq_len, num_heads, head_dim)
        ctx = (att @ v).transpose(1, 2)
        
        # Concatenate heads to get (bs, seq_len, dim) & make it contiguous in memory
        #ctx = ctx.contiguous().view(bs, seq_len, self.dim)
        ctx = ctx.reshape(bs, seq_len, self.dim)
        ctx = self.proj(ctx)

        return ctx

# x = torch.randn(3, 4)
# batch_x = torch.stack((x, x), dim=0) # Create a batch of input
# din = batch_x.shape[2] # Input dim
# seq_len = 10 # Max ctx length supported
# dim = 16 # dim of att layer embeddings
# num_heads = 8
# num_kv_groups = 2 # i.e. 4 q heads share 1 k,v 
# gqa_layer = GQARoPE(din, dim, seq_len, 0.1, num_heads, num_kv_groups, theta_base=10000.0)
# ctx = gqa_layer(batch_x) 
# total_params = sum(p.numel() for p in gqa_layer.parameters())
# print(f"Number of parameters: {total_params:,}")
# print(batch_x.shape)
# print(ctx.shape, ctx)  # bs, seqlen, dim

# ----------
# Tokenizations
# ----------