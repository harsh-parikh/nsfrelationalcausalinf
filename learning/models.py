import torch


class PhiNet(torch.nn.Module):
    def __init__(self, direct, summaries):
        super(PhiNet, self).__init__()

        self.n = direct + len(summaries)

        self.final = torch.nn.Linear(self.n, 1)

        self.summarize = torch.nn.ModuleList(
            [ torch.nn.Linear(s, 1) for s in summaries ]
        )

    def forward(self, direct, *args):
        if len(args) + len(direct) != self.n:
            raise ValueError(f"PhiNet received {len(args) + len(direct)} arguments, "
                             f"expected {self.n}")

        intermediates = direct.squeeze().unsqueeze(-1)
        for inp, net in zip(args, self.summarize):
            intermediates = torch.cat((intermediates, net(inp)), dim=-1)

        return self.final(intermediates)
