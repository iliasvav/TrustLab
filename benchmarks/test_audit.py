from pathlib import Path

import pandas as pd

from src.trustlab.audit import inspect_target_associations


DATA_DIR = Path("benchmarks/data/leakage_case")


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")

    associations = inspect_target_associations(train)

    print("\nTOP FEATURE-TARGET ASSOCIATIONS\n")

    for result in associations:
        correlation = (
            f"{result.correlation:.3f}"
            if result.correlation is not None
            else "N/A"
        )

        print(
            f"{result.feature:20} "
            f"corr={correlation:>7} "
            f"MI={result.mutual_information:.3f}"
        )


if __name__ == "__main__":
    main()
