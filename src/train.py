import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    Subset
)

from dataset import TripletFaceDataset

from model import FaceEmbeddingModel


# -----------------------------------
# DEVICE
# -----------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"\nUsing Device: {DEVICE}\n")


# -----------------------------------
# CONFIG
# -----------------------------------

BATCH_SIZE = 8

EPOCHS = 2

LEARNING_RATE = 0.001

NUM_WORKERS = 0


# -----------------------------------
# DATASET
# -----------------------------------

dataset = TripletFaceDataset(

    csv_file="data/train_balanced.csv",

    image_dir="data/UTKFace"
)


# -----------------------------------
# SMALLER SUBSET FOR SPEED
# -----------------------------------

MAX_SAMPLES = min(
    3000,
    len(dataset)
)

subset_indices = list(
    range(MAX_SAMPLES)
)

dataset = Subset(
    dataset,
    subset_indices
)


# -----------------------------------
# DATALOADER
# -----------------------------------

loader = DataLoader(

    dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()
)


# -----------------------------------
# MODEL
# -----------------------------------

model = FaceEmbeddingModel().to(DEVICE)


# -----------------------------------
# LOSS
# -----------------------------------

criterion = nn.TripletMarginLoss(
    margin=1.0
)


# -----------------------------------
# OPTIMIZER
# -----------------------------------

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE
)


# -----------------------------------
# MIXED PRECISION
# -----------------------------------

scaler = torch.cuda.amp.GradScaler("cuda",
    enabled=torch.cuda.is_available()
)


# -----------------------------------
# TRAINING LOOP
# -----------------------------------

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0


    for anchor, positive, negative in loader:

        anchor = anchor.to(
            DEVICE,
            non_blocking=True
        )

        positive = positive.to(
            DEVICE,
            non_blocking=True
        )

        negative = negative.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad()


        with torch.cuda.amp.autocast(
            enabled=torch.cuda.is_available()
        ):

            anchor_embedding = model(anchor)

            positive_embedding = model(
                positive
            )

            negative_embedding = model(
                negative
            )

            loss = criterion(

                anchor_embedding,

                positive_embedding,

                negative_embedding
            )


        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()


        total_loss += loss.item()


    avg_loss = total_loss / len(loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {avg_loss:.4f}"
    )


# -----------------------------------
# SAVE MODEL
# -----------------------------------

SAVE_PATH = (
    "artifacts/model_mitigated.pt"
)

torch.save(

    model.state_dict(),

    SAVE_PATH
)


print(
    f"\nModel saved successfully: "
    f"{SAVE_PATH}\n"
)