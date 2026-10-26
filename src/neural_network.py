from torch import nn

# If someone had an accelerator (e.g. Nvidia GPU), then this would have been helpful.
# If available, someone could move the whole network there: model = NeuralNetwork().to(<target_device>)
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"Using {device} device")

# As we have 12 inputs (left_knee_angle, right_knee_angle, ...) and 6 label-classes (0,1,2,3,4,5),
# we should have input 12 and output 6.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten = nn.Flatten() if we had 2D arrays (e.g. images), this would have been helpful.
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(), # This is essentially max(0, x) - the weight dies if it is smaller than zero.
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6),
        )

    # TODO: find out what this does
    def forward(self, x):
        # x = self.flatten(x)
        return self.linear_relu_stack(x)