import json


with open(
    "results/initial_audit.json",
    "r"
) as f:

    audit_results = json.load(f)


groups = [

    key for key in audit_results.keys()

    if key != "overall"
]


worst_group = groups[0]

max_far = audit_results[
    worst_group
]["far"]


for group in groups:

    current_far = audit_results[
        group
    ]["far"]

    if current_far > max_far:

        max_far = current_far

        worst_group = group


analysis = {

    "most_biased_pairing": {

        "group_1": worst_group,

        "group_2": "overall",

        "metric": "far_disparity",

        "value": round(max_far, 4)
    },

    "hypothesized_causes": {

        "data_level":
        "Some demographic groups may be underrepresented in the dataset, causing uneven feature learning and poor generalization.",

        "model_level":
        "The pretrained CNN backbone may inherit demographic biases from large-scale pretraining datasets."
    }
}


with open(
    "results/analysis.json",
    "w"
) as f:

    json.dump(
        analysis,
        f,
        indent=4
    )


print("\nBias analysis completed\n")

print(
    json.dumps(
        analysis,
        indent=4
    )
)