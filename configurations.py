
class self_attend_configs():
    max_blocksize: int = 512
    hidden_dim: int = 768
    nhead: int  = 8
    use_flash_attn: True
    dropout: float = 0.1
    bias : bool = True

