from torch import nn

class LinearRegressionClassifier(nn.Module):
    def __init__(self):
        super(LinearRegressionClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)
