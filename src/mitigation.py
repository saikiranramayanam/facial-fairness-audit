import pandas as pd


df = pd.read_csv(
    "data/train.csv"
)


balanced_groups = []


groups = df.groupby(
    ["gender", "race"]
)


min_group_size = min(

    len(group)

    for _, group in groups
)


for _, group in groups:

    sampled_group = group.sample(
        min_group_size,
        random_state=42
    )

    balanced_groups.append(
        sampled_group
    )


balanced_df = pd.concat(
    balanced_groups
)


balanced_df.to_csv(
    "data/train_balanced.csv",
    index=False
)


print(
    "\nBalanced dataset created\n"
)

print(
    f"Balanced size: {len(balanced_df)}"
)