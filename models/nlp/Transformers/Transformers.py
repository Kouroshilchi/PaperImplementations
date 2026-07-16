import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
    def forward(self, Q, K, V, mask=None):
        dot_product_ = Q @ K.transpose(-2,-1)
        scores = dot_product_ / self.d_k ** 0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = torch.softmax(scores, dim=-1)
        output = scores @ V
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.Q_T = nn.Linear(d_model, d_model, bias=False)
        self.V_T = nn.Linear(d_model, d_model, bias=False)
        self.K_T = nn.Linear(d_model, d_model, bias=False)
        self.O_T = nn.Linear(d_model, d_model, bias=False)
        self.scaled_attention = ScaledDotProductAttention(self.d_k)
    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        Q = self.Q_T(x)
        V = self.V_T(x)
        K = self.K_T(x)
        Q = Q.view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        output = self.scaled_attention(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.O_T(output)

class CrossAttention(nn.Module):
    def __init__(self, heads, d_model):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.Q_T = nn.Linear(d_model, d_model, bias=False)
        self.V_T = nn.Linear(d_model, d_model, bias=False)
        self.K_T = nn.Linear(d_model, d_model, bias=False)
        self.O_T = nn.Linear(d_model, d_model, bias=False)
        self.scaled_attention = ScaledDotProductAttention(self.d_k)
    def forward(self, x_q, x_kv, mask=None):
        batch_size = x_q.shape[0]
        Q = self.Q_T(x_q)
        V = self.V_T(x_kv)
        K = self.K_T(x_kv)
        Q = Q.view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        output = self.scaled_attention(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.O_T(output)

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_seq_length):
        super().__init__()
        pe = torch.zeros(max_seq_length, dim_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * 
                                (-math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)      
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len].to(x.device)

class FeedForward(nn.Module):
    def __init__(self, dim_model, dim_ff):
        super().__init__()
        self.linear1 = nn.Linear(dim_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, dim_model)
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_model: int = 128, num_heads: int = 8, 
                 ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        
        if ff_dim is None:
            ff_dim = dim_model * 4
        self.MultiHeadAttention = MultiHeadAttention(num_heads, dim_model)
        self.layernorm_mha = nn.LayerNorm(dim_model)
        self.ffl = FeedForward(dim_model, ff_dim)
        self.layernorm_ffl = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layernorm_mha(x + self.dropout(self.MultiHeadAttention(x)))
        x = self.layernorm_ffl(x + self.dropout(self.ffl(x)))
        return x        

class TransformerEncoder(nn.Module):
    def __init__(self, dim_model=128, num_layers=6, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.pos_encoder = PositionalEncoding(dim_model, max_seq_length=512)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim_model, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)            
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim_model: int = 128, num_heads: int = 8, 
                 ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        
        if ff_dim is None:
            ff_dim = dim_model * 4
        self.masked_multihead_attn = MultiHeadAttention(num_heads, dim_model)
        self.layernorm_mha = nn.LayerNorm(dim_model)
        self.cross_attention = CrossAttention(num_heads, dim_model)
        self.layernorm_cross = nn.LayerNorm(dim_model)
        self.ffl = FeedForward(dim_model, ff_dim)
        self.layernorm_ffl = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_kv, tgt_mask=None, src_mask=None):
        x = self.layernorm_mha(x + self.dropout(self.masked_multihead_attn(x, tgt_mask)))
        x = self.layernorm_cross(x + self.dropout(self.cross_attention(x, x_kv, src_mask)))
        x = self.layernorm_ffl(x + self.dropout(self.ffl(x)))
        return x        

class TransformerDecoder(nn.Module):
    def __init__(self, dim_model=128, num_layers=6, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.pos_encoder = PositionalEncoding(dim_model, max_seq_length=512)
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(dim_model, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.pos_encoder(x)                  
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

if __name__=="__main__":
    Transformer = TransformerDecoder(128, 6, 8, 0.1).to('cuda')
    embedd = torch.randn([10,10,128], device='cuda')
    with torch.no_grad():
        output = Transformer(embedd, embedd)
    print(output.shape)