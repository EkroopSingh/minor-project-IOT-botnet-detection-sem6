import pandas as pd
import os

DATASET_PATH = "dataset"
OUTPUT_PATH = "outputs/final_dataset.csv"

os.makedirs("outputs", exist_ok=True)

dataframes = []

for file in os.listdir(DATASET_PATH):
    if file.endswith(".csv"):
        file_path = os.path.join(DATASET_PATH, file)
        print(f"Loading {file}...")
        
        df = pd.read_csv(file_path)

        if "benign" in file.lower():
            df["label"] = 0
        else:
            df["label"] = 1

        dataframes.append(df)

final_df = pd.concat(dataframes, ignore_index=True)

final_df.to_csv(OUTPUT_PATH, index=False)

print("✅ Dataset created successfully!")
print("Shape:", final_df.shape)
