import torch
from torch import nn

class LinearRegressionSum(nn.Module):
    def __init__(self):
        super(LinearRegressionSum, self).__init__()
        self.input_sk = nn.Linear(1, 1)

        self.output_sk = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, v):
        mapped_v = self.input_sk(v.unsqueeze(dim=-1))\
            .sum().unsqueeze(dim=-1)
        count = len(mapped_v)
        temp = torch.cat((mapped_v, torch.Tensor([count])), dim=-1)
        return self.output_sk(temp)
