import torch
from torch import nn

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


# as we have 12 inputs (left_knee_angle, right_knee_angle, ...) and 4 label-classes (0,1,3,4),
# we should have input 12 and output 4
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() #TODO: find out what this does
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(), # this is essentially max(0, x) - the weight dies if it is smaller than zero
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

    # TODO: find out what this does
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()