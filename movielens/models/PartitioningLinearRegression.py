import torch
from torch import nn

class PartitioningLinearRegression(nn.Module):
    """
        Turn the input vector into blocks, run linear regression over them.
        Sum the intermediate results and run linear regression again.
    """
    def __init__(self, block_size):
        super(PartitioningLinearRegression, self).__init__()
        self.block_size = block_size
        self.input_sk = nn.Linear(block_size, 1)

        self.output_sk = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, v):
        mapped_v = self.input_sk(v).sum(dim=0)
        count = len(v) * self.block_size
        temp = torch.cat((mapped_v, torch.Tensor([count])), dim=-1)
        return self.output_sk(temp)
