import os
import json

import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F

from torchvision import transforms

from sklearn.metrics import confusion_matrix

from model import FaceEmbeddingModel


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

THRESHOLD = 0.85

AUDIT_CSV = "data/audit.csv"

IMAGE_DIR = "data/UTKFace"

OUTPUT_JSON = (
    "results/mitigated_audit.json"
)


df = pd.read_csv(AUDIT_CSV)


transform = transforms.Compose([

    transforms.Resize((64, 64)),

    transforms.ToTensor()
])


model = FaceEmbeddingModel().to(DEVICE)

model.load_state_dict(
    torch.load(
        "artifacts/model_mitigated.pt",
        map_location=DEVICE
    )
)

model.eval()


embedding_cache = {}


# -----------------------------------
# EMBEDDING
# -----------------------------------

def get_embedding(image_name):

    if image_name in embedding_cache:

        return embedding_cache[image_name]

    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    image = Image.open(image_path).convert(
        "RGB"
    )

    image = transform(image)

    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        embedding = model(image)

    embedding_cache[image_name] = embedding

    return embedding


# -----------------------------------
# COSINE SIMILARITY
# -----------------------------------

def cosine_similarity(
    embedding1,
    embedding2
):

    similarity = F.cosine_similarity(

        embedding1,

        embedding2
    )

    return similarity.item()


# -----------------------------------
# CLASSIFICATION
# -----------------------------------

def classify_pair(
    similarity,
    threshold
):

    if similarity >= threshold:

        return 1

    return 0


# -----------------------------------
# AGE GROUP
# -----------------------------------

def get_age_group(age):

    if age <= 19:

        return "0-19"

    elif age <= 39:

        return "20-39"

    elif age <= 59:

        return "40-59"

    return "60+"


# -----------------------------------
# SKIN TONE
# -----------------------------------

def get_skin_tone(race):

    if race == 0:

        return "Light"

    elif race in [2, 3]:

        return "Medium"

    return "Dark"


results = {}

overall_y_true = []

overall_y_pred = []


# -----------------------------------
# AUDIT LOOP
# -----------------------------------

for idx in range(len(df)):

    anchor_row = df.iloc[idx]

    anchor_image = anchor_row["image"]

    anchor_gender = (
        "Male"
        if anchor_row["gender"] == 0
        else "Female"
    )

    anchor_age_group = get_age_group(
        anchor_row["age"]
    )

    anchor_skin = get_skin_tone(
        anchor_row["race"]
    )

    subgroup = (
        f"{anchor_gender}_"
        f"{anchor_age_group}_"
        f"{anchor_skin}"
    )

    if subgroup not in results:

        results[subgroup] = {

            "false_accepts": 0,

            "false_rejects": 0,

            "positive_total": 0,

            "negative_total": 0
        }

    # POSITIVE PAIR
    positive_df = df[

        (df["gender"] == anchor_row["gender"]) &

        (df["race"] == anchor_row["race"]) &

        (df.index != idx)
    ]

    if len(positive_df) == 0:
        continue

    positive_row = positive_df.sample().iloc[0]

    # NEGATIVE PAIR
    negative_df = df[

        (df["gender"] != anchor_row["gender"]) |

        (df["race"] != anchor_row["race"])
    ]

    negative_row = negative_df.sample().iloc[0]

    # EMBEDDINGS
    anchor_embedding = get_embedding(
        anchor_image
    )

    positive_embedding = get_embedding(
        positive_row["image"]
    )

    negative_embedding = get_embedding(
        negative_row["image"]
    )

    # SIMILARITY
    positive_similarity = cosine_similarity(

        anchor_embedding,

        positive_embedding
    )

    negative_similarity = cosine_similarity(

        anchor_embedding,

        negative_embedding
    )

    # FAIRNESS MITIGATION
    adjusted_threshold = THRESHOLD

    if anchor_skin == "Dark":

        adjusted_threshold -= 0.03

    if anchor_age_group == "60+":

        adjusted_threshold -= 0.02

    # POSITIVE PREDICTION
    positive_prediction = classify_pair(

        positive_similarity,

        adjusted_threshold
    )

    results[subgroup][
        "positive_total"
    ] += 1

    overall_y_true.append(1)

    overall_y_pred.append(
        positive_prediction
    )

    if positive_prediction == 0:

        results[subgroup][
            "false_rejects"
        ] += 1

    # NEGATIVE PREDICTION
    negative_prediction = classify_pair(

        negative_similarity,

        adjusted_threshold
    )

    results[subgroup][
        "negative_total"
    ] += 1

    overall_y_true.append(0)

    overall_y_pred.append(
        negative_prediction
    )

    if negative_prediction == 1:

        results[subgroup][
            "false_accepts"
        ] += 1


# -----------------------------------
# FINAL RESULTS
# -----------------------------------

final_results = {}


for subgroup, metrics in results.items():

    far = (

        metrics["false_accepts"] /

        metrics["negative_total"]
    )

    frr = (

        metrics["false_rejects"] /

        metrics["positive_total"]
    )

    final_results[subgroup] = {

        "far": round(far, 4),

        "frr": round(frr, 4)
    }


# -----------------------------------
# OVERALL METRICS
# -----------------------------------

tn, fp, fn, tp = confusion_matrix(

    overall_y_true,

    overall_y_pred

).ravel()

overall_far = fp / (fp + tn)

overall_frr = fn / (fn + tp)

final_results["overall"] = {

    "far": round(overall_far, 4),

    "frr": round(overall_frr, 4)
}


# -----------------------------------
# SAVE JSON
# -----------------------------------

with open(OUTPUT_JSON, "w") as f:

    json.dump(
        final_results,
        f,
        indent=4
    )


print("\nMitigated audit completed\n")

print(
    json.dumps(
        final_results,
        indent=4
    )
)