import torch
from torch import nn


class PartitioningNN(nn.Module):
    def __init__(self, block_size):
        super(PartitioningNN, self).__init__()
        self.block_size = block_size
        k = 19 * self.block_size
        self.input_sk = nn.Sequential(
            nn.Linear(self.block_size, (k + self.block_size) // 2),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear((k + self.block_size) // 2, k),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(k, k)
        )

        self.output_sk = nn.Sequential(
            nn.Linear(k + 1, k * 5),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(k * 5, k * 5),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(k * 5, 1),
            nn.Sigmoid()
        )

    def forward_all(self, users):
        outputs = []
        for user in users:

            outputs.append(self.forward(user.reshape((-1, self.block_size))))
        return torch.stack(outputs).squeeze()

    def forward(self, user):
        mapped_v = self.input_sk(user).sum(dim=0)
        count = len(user) * self.block_size
        temp = torch.cat((mapped_v, torch.Tensor([count])), dim=-1)
        return self.output_sk(temp)
