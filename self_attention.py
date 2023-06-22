import torch
import torch.nn.functional as F
import math
import torch.nn as nn

class self_attend_configs():
    max_blocksize: int = 512
    hidden_dim: int = 768
    nhead: int  = 8
    is_causal: bool = False
    use_flash_attn: True
    dropout: float = 0.1
    bias : bool = True

class SelfAttend(nn.Module):
    def __init__(self,configs):
        super(self,SelfAttend).__init__()
        self.configs = configs

        #assert (configs['hidden_dim'] % configs['nhead']), 'token hidden dimension and number of heads must be devisible'

        self.query = nn.Linear(self.configs.hidden_dim,self.configs.hidden_dim)
        self.key = nn.Linear(self.configs.hidden_dim,self.configs.hidden_dim)
        self.value  = nn.Linear(self.configs.hidden_dim,self.configs.hidden_dim)

        self.c_proj = nn.Linear(self.configs.hidden_dim, self.configs.hidden_dim,bias=self.configs.bias)
        self.resid_dropout = nn.Dropout(self.configs.dropout)
        self.attn_dropout = nn.Dropout(0.1)
        
        if configs.is_causal and not self.configs.use_flash:
            self.register_buffer('attn_mask',torch.tril(torch.ones(1, 1, self.configs.max_blocksize,self.configs.max_blocksize,dtype=bool)))

        self.use_flash_attn = configs.use_flash_attn

    def forward(self,x,padding_mask: bool = None):
        B, T, hd = x.shape

        q = self.query(x).view(B, T, self.configs.nhead, -1).permute(0, 2, 1, 3)  # (B, nhead, T, hs)
        k = self.key(x).view(B, T, self.configs.nhead, -1).permute(0, 2, 1, 3)    # (B, nhead, T, hs)
        v = self.value(x).view(B, T, self.configs.nhead, -1).permute(0, 2, 1, 3)  # (B, nhead, T, hs)

        if self.configs.use_flash_attn:

            if self.configs.is_causal:

                with torch.backends.cuda.sdp_kernel(enable_math=False): # use the most efficient implementation fused kernel
                    output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)

            else:
                with torch.backends.cuda.sdp_kernel(enable_math=False):
                    output = F.scaled_dot_product_attention(q, k, v,attn_mask = None, is_causal=False)

        else:
            
            scaler = (1.0 / math.sqrt(k.shape[-1]))
            attn_weight = (q @ k.transpose(2,3)) * scaler

            if self.configs.is_causal:

                masked_weights = attn_weight.masked_fill(self.attn_mask[:,:,:T,:T]==0, float('-inf'))
                if padding_mask:
                    masked_weights.masked_fill(padding_mask == 0, float('-inf'))

                norm_weights = F.softmax(masked_weights,dim=-1)
                norm_weights = self.attn_dropout(norm_weights)

                output = norm_weights @ v
            else:

                if padding_mask:
                    attn_weight.masked_fill(padding_mask == 0, float('-inf'))

                norm_weights = F.softmax(attn_weight,dim=-1)

                output = norm_weights @ v
                

        return output
