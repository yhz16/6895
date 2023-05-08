import pandas as pd
import numpy as np

data_path = "/home/yhz/Downloads/02-14-2018.csv"
data = pd.read_csv(data_path)

# Select 100 random records
sample_data = data.sample(n=100, random_state=42)

# Save the sample data to test.csv
sample_data.to_csv("test.csv", index=False)

# Calculate the percentage of each label
label_counts = sample_data['Label'].value_counts(normalize=True) * 100

print("Percentage of each label in the test.csv:")
print(label_counts)
