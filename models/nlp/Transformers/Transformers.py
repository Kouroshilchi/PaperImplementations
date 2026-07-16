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
        scores = dot_product_ / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32, device=Q.device))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = torch.softmax(scores, dim=-1)
        output = scores @ V
        return output