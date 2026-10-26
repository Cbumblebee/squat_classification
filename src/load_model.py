import torch, litert_torch
import fetch_data
from neural_network import NeuralNetwork


class WithScaling(torch.nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


train_ds = fetch_data.CustomSquatDataset("../data/squat_dataset/squat_features_augmented.csv")
scaler = train_ds.scaler
print(list(scaler.feature_names_in_))  # the feature order the app must follow

model = NeuralNetwork()
model.load_state_dict(torch.load("../models/model_adam_lr0.01.pth", map_location="cpu"))  # best model
model.eval()

export_model = WithScaling(model, scaler.mean_, scaler.scale_).eval()

# Sanity check: wrapped model on raw input == old pipeline
raw = torch.tensor(scaler.inverse_transform(train_ds.feature_values[:5]), dtype=torch.float32)
old = model(torch.tensor(train_ds.feature_values[:5], dtype=torch.float32))
new = export_model(raw)
print(torch.allclose(old, new, atol=1e-4))  # should print True

# Export
sample = torch.randn(1, len(scaler.mean_))
litert_torch.convert(export_model, (sample,)).export("squat_detector.tflite")