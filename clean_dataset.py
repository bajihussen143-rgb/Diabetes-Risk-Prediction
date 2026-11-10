
import pandas as pd
from pathlib import Path


# Dataset paths
INPUT_PATH = Path("data/diabetes.csv")
OUTPUT_PATH = Path("data/diabetes_cleaned.csv")


# Columns where zero represents missing information
INVALID_ZERO_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]


def load_dataset():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print("Dataset loaded successfully!")

    return df


def inspect_zero_values(df):

    print("\nBefore cleaning:")

    for column in INVALID_ZERO_COLUMNS:

        if column not in df.columns:
            raise ValueError(
                f"Required column missing: {column}"
            )

        zero_count = (df[column] == 0).sum()

        print(f"{column}: {zero_count} zero values")


def replace_invalid_zeros(df):

    print("\nReplacing invalid zeros with NaN...")

    for column in INVALID_ZERO_COLUMNS:

        df[column] = df[column].replace(0, float("nan"))

    return df


def inspect_missing_values(df):

    print("\nMissing values after cleaning:")

    print(df.isnull().sum())

    print("\nTotal missing values:")

    print(df.isnull().sum().sum())


def save_cleaned_dataset(df):

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nCleaned dataset saved successfully!")

    print("Saved location:", OUTPUT_PATH)


def main():

    try:

        df = load_dataset()

        inspect_zero_values(df)

        df = replace_invalid_zeros(df)

        inspect_missing_values(df)

        save_cleaned_dataset(df)


    except Exception as error:

        print("\nERROR:", error)


if __name__ == "__main__":
    main()
