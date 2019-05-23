import torch
from torch import nn

# embedding dimension
j = 19

class EmbeddingSum(nn.Module):
    def __init__(self):
        super(EmbeddingSum, self).__init__()

        self.e = nn.Embedding(2000, j)
        self.s = nn.Sigmoid()

    def forward_all(self, movies):
        e_outputs = []
        for m in movies:
            e_outputs.append(self.forward(m))
        return torch.stack(e_outputs).squeeze()

    def forward(self, movies):
        lookup = self.e((movies).long())
        return self.s(lookup.mean())
