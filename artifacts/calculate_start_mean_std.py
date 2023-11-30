import json

import numpy as np
from datasets import load_dataset


def calculate_mean_std():
    ds = load_dataset("JasiekKaczmarczyk/maestro-v1-sustain-masked", split="train")

    mean_start = np.mean(ds["start"])
    std_start = np.std(ds["start"])

    features = {
        "mean_start": mean_start,
        "std_start": std_start,
    }

    with open("artifacts/time_features.json", "x") as f:
        json.dump(features, f)
        f.close()


if __name__ == "__main__":
    calculate_mean_std()
