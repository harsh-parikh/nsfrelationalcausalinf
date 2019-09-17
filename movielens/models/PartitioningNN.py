import torch
from torch import nn


class PartitioningNN(nn.Module):
    def __init__(self, block_size):
        super(PartitioningNN, self).__init__()
        self.block_size = block_size
        k = 45
        self.input_sk = nn.Sequential(
            nn.Linear(self.block_size, k),
            nn.ReLU(),
            nn.Linear(k, k),
            nn.ReLU(),
            nn.Linear(k, k),
            nn.ReLU()
        )

        self.output_sk = nn.Sequential(
            nn.Linear(k, k),
            nn.ReLU(),
            nn.Linear(k, k),
            nn.ReLU(),
            nn.Linear(k, 1),
            nn.Sigmoid()
        )

    def forward_all(self, users):
        outputs = []
        for user in users:
            outputs.append(self.forward(user.reshape((-1, self.block_size))))
        return torch.stack(outputs).squeeze()

    def forward(self, user):
        #count = len(user) * self.block_size
        mapped_v = self.input_sk(user).sum(dim=0)
        return self.output_sk(mapped_v)
