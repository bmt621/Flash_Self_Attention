import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import logging

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias,eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias,eps=self.eps)
    

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.hidden_dim, 4 * config.hidden_dim, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.hidden_dim, config.hidden_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

class SelfAttend(nn.Module):
    def __init__(self,configs):
        super(SelfAttend,self).__init__()

        self.configs = configs
        logging.basicConfig(level=logging.INFO)  
        self.logger = logging.getLogger(__name__)  # Create a logger instance

        self.query = nn.Linear(self.configs.hidden_dim,self.configs.hidden_dim)
        self.key = nn.Linear(self.configs.hidden_dim,self.configs.hidden_dim)
        self.value  = nn.Linear(self.configs.hidden_dim,self.configs.hidden_dim)

        self.proj = nn.Linear(self.configs.hidden_dim, self.configs.hidden_dim,bias=self.configs.bias)
        self.proj_dropout = nn.Dropout(self.configs.dropout)
        self.attn_dropout = nn.Dropout(0.1)
        

        self.use_flash_attn = configs.use_flash_attn

    def get_qkv(self,xq,xk,xv):
        assert xq.shape[-1] == xk.shape[-1] == xv.shape[-1], self.logger.info('shape of query, key and value embedding dimension must be thesame')

        B, qT, hd = xq.shape
        _, kT, _ = xk.shape
        _, vT, _ = xv.shape

        assert kT == vT, self.logger.info('query length and values length should be the same')
        
        q = self.query(xq).view(B, qT, self.configs.nhead, -1).permute(0, 2, 1, 3)
        k = self.key(xk).view(B, kT, self.configs.nhead, -1).permute(0, 2, 1, 3)
        v = self.value(xv).view(B, vT, self.configs.nhead, -1).permute(0, 2, 1, 3)

        return q, k, v
    

    def forward(self,q, k, v,padding_mask = None, is_causal = False):
        
        if is_causal and not self.configs.use_flash_attn:
            self.register_buffer('attn_mask',torch.tril(torch.ones(1, 1, self.configs.max_blocksize,self.configs.max_blocksize,dtype=bool)))

        if self.configs.use_flash_attn:
            self.logger.info("This implementation of flash attention does not support src_key_padding_mask or tgt_key_padding_mask")
            
            if is_causal:
                with torch.backends.cuda.sdp_kernel(enable_math=False): # use the most efficient implementation fused kernel
                    output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
            else:
                with torch.backends.cuda.sdp_kernel(enable_math=False):
                    output = F.scaled_dot_product_attention(q, k, v,attn_mask = None, is_causal=False)

        else:
            
            scaler = (1.0 / math.sqrt(k.shape[-1]))
            attn_weight = (q @ k.transpose(2,3)) * scaler

            if is_causal:
                masked_weights = attn_weight.masked_fill(self.attn_mask[:,:,:q.shape[2],:q.shape[2]]==0, float('-inf'))

                if padding_mask:
                    masked_weights.masked_fill(padding_mask == 0, float('-inf'))

                norm_weights = F.softmax(masked_weights,dim=-1)
                norm_weights = self.attn_dropout(norm_weights)

                output = norm_weights @ v # (B, nh, T, hs)

            else:
                
                if padding_mask:
                    attn_weight.masked_fill(padding_mask == 0, float('-inf'))

                norm_weights = F.softmax(attn_weight,dim=-1)

                output = norm_weights @ v # (B, nh, T, hs)
                
            
        output = output.permute(0, 2, 1, 3).flatten(start_dim=2)    
        output = self.proj_dropout(self.proj(output))

        return output

class EncoderBlock(nn.Module):

    def __init__(self, config):
        super(EncoderBlock, self).__init__()

        self.ln_1 = LayerNorm(config.hidden_dim, bias=config.bias)
        self.attn = SelfAttend(config)

        self.ln_2 = LayerNorm(config.hidden_dim, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x,src_padding_mask = None):

        x = self.ln_1(x)
        q, k, v = self.attn.get_qkv(x,x,x)

        x = x + self.attn(q, k, v, src_padding_mask)
        x = x + self.mlp(self.ln_2(x))

        return x
    

class DecoderBlock(nn.Module):

    def __init__(self,config):
        super(DecoderBlock, self).__init__()

        self.ln_1 = LayerNorm(config.hidden_dim, bias=config.bias)
        self.attn = SelfAttend(config)

        self.ln_2 = LayerNorm(config.hidden_dim, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, mem, tgt_padding_mask = None):
        x = self.ln_1(x)
        q, k, v = self.attn.get_qkv(x, x, x)

        new_state_of_x = self.attn(q,k,v, is_causal=True)
        q, k, v = self.attn.get_qkv(new_state_of_x, mem, mem)

        x = x + self.attn(q,k,v,is_causal=False, padding_mask = tgt_padding_mask)
        x = x + self.mlp(self.ln_2(x))

        return x