import json


with open(
    "results/initial_audit.json",
    "r"
) as f:

    initial_audit = json.load(f)


with open(
    "results/mitigated_audit.json",
    "r"
) as f:

    mitigated_audit = json.load(f)


initial_far = initial_audit[
    "overall"
]["far"]

initial_frr = initial_audit[
    "overall"
]["frr"]


mitigated_far = mitigated_audit[
    "overall"
]["far"]

mitigated_frr = mitigated_audit[
    "overall"
]["frr"]


initial_accuracy = 1 - (
    (initial_far + initial_frr) / 2
)

mitigated_accuracy = 1 - (
    (mitigated_far + mitigated_frr) / 2
)


metrics = {

    "initial_model": {

        "accuracy": round(
            initial_accuracy,
            4
        ),

        "threshold": 0.85
    },

    "mitigated_model": {

        "accuracy": round(
            mitigated_accuracy,
            4
        ),

        "threshold": 0.80
    }
}


with open(
    "results/overall_metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


print("\nOverall metrics generated\n")

print(
    json.dumps(
        metrics,
        indent=4
    )
)