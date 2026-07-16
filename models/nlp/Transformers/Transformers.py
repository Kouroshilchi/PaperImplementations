import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

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
        

if __name__=="__main__":
    attention = MultiHeadAttention(16, 128).to('cuda')
    embedd = torch.randn([10,1,128], device='cuda')
    with torch.no_grad():
        output = attention(embedd)
    print(output.shape)