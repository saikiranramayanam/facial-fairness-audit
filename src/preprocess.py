import os
import pandas as pd

DATASET_PATH = "data/UTKFace"

data = []

for image_name in os.listdir(DATASET_PATH):

    try:
        parts = image_name.split("_")

        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])

        data.append({
            "image": image_name,
            "age": age,
            "gender": gender,
            "race": race
        })

    except:
        continue

df = pd.DataFrame(data)

print(df.head())

df.to_csv("data/metadata.csv", index=False)

print("Metadata saved successfully")