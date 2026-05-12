import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/metadata.csv")

train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42
)

val_df, audit_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42
)

train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
audit_df.to_csv("data/audit.csv", index=False)

print("Dataset split completed")

print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Audit: {len(audit_df)}")