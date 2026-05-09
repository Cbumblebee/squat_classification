from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
from torch.utils.data import random_split

class CustomSquatDataset(Dataset):
    def __init__(self, csv_data_file):
        # CITE: extracting features and values from the README.md from the kaggle project
        df = pd.read_csv(csv_data_file)
        # Separate features and labels
        # drop some unneeded metadata
        features = df.drop(["video_file", "frame", "label"], axis=1)
        labels = df["label"]
        # Normalization: putting degrees and other formats into perspective of a [0,1] range
        # CITE: I have found this source: https://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range
        self.scaler = StandardScaler()
        self.feature_values = self.scaler.fit_transform(features)
        self.label_values = labels.values

    def __len__(self):
        return len(self.feature_values)

    def __getitem__(self, idx):
        # PyTorch wants float32 for features and long for classification labels
        features = torch.tensor(self.feature_values[idx], dtype=torch.float32)
        label = torch.tensor(self.label_values[idx], dtype=torch.long)
        return features, label


# Train your model using X, y
dataset = CustomSquatDataset("../data/squat_dataset/squat_features_augmented.csv")
# since I have only one dataset of this format, I need to split it into training and testing dataset
training_len = int(len(dataset) * 0.8) # 80% of the dataset is used for training
testing_len = len(dataset) - training_len #20% is used for testing
train_data, test_data = random_split(dataset, [training_len, testing_len])
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Sanity Check: Look at one batch
# for features, labels in train_loader:
#     print(f"Feature batch shape: {features.shape}") # Should be [32, 12]
#     print(f"Label batch shape: {labels.shape}")     # Should be [32]
#     break

