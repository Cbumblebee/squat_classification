# z-score normalization
left_knee_angle (120.25) vs. left_knee_lateral (0.077)


# This weekend (8th til 10th May):
3. Saturday Morning Goal: The "Silent" Model
Before your meeting, try to get the code to run without errors. You don't even need to check accuracy yet. Just verify that:

The CSV loads.

The StandardScaler doesn't crash.

The DataLoader can provide one batch of data.

Pro-tip for an Arch user: Use IPython or a Jupyter Notebook (if you have it in your .venv) for this exploratory phase. It lets you inspect the tensors easily without re-running the whole script every time.

Wait... one quick check: Does your dataset have any empty rows at the very end of the CSV? Sometimes Kaggle exports have a trailing newline that makes pandas think there's a row of NaNs. If your script crashes immediately, run df = df.dropna() after loading.