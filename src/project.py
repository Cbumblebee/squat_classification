# code from the README.md from the kaggle project
import pandas as pd

# Load the dataset
df = pd.read_csv("squat_features_augmented.csv")

# Separate features and labels
# drop some unneeded metadata
X = df.drop("video_file", "frame", "label", axis=1)
y = df["label"]

# Train your model using X, y