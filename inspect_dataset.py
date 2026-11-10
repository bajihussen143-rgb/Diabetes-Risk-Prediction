
import pandas as pd
from pathlib import Path


# ----------------------------------------
# STEP 1: Define the dataset file location
# ----------------------------------------

DATASET_PATH = Path("data/diabetes.csv")


# ----------------------------------------
# STEP 2: Load the dataset
# ----------------------------------------

def load_dataset():
    """Load the diabetes CSV dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print("\nDataset loaded successfully!")

    return df


# ----------------------------------------
# STEP 3: Display basic dataset information
# ----------------------------------------

def inspect_basic_information(df):

    print("\n========== DATASET OVERVIEW ==========")

    print("\nNumber of rows and columns:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 records:")
    print(df.head())

    print("\nLast 5 records:")
    print(df.tail())

    print("\nDataset information:")
    df.info()


# ----------------------------------------
# STEP 4: Check missing values
# ----------------------------------------

def inspect_missing_values(df):

    print("\n========== MISSING VALUES ==========")

    missing_values = df.isnull().sum()

    print(missing_values)

    print("\nTotal missing values:")
    print(missing_values.sum())


# ----------------------------------------
# STEP 5: Check zero values
# ----------------------------------------

def inspect_zero_values(df):

    print("\n========== ZERO VALUES ==========")

    medical_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    for column in medical_columns:

        if column in df.columns:
            zero_count = (df[column] == 0).sum()

            print(f"{column}: {zero_count} zero values")


# ----------------------------------------
# STEP 6: Check duplicate records
# ----------------------------------------

def inspect_duplicates(df):

    print("\n========== DUPLICATE RECORDS ==========")

    duplicate_count = df.duplicated().sum()

    print("Number of duplicate records:", duplicate_count)


# ----------------------------------------
# STEP 7: Display statistical information
# ----------------------------------------

def inspect_statistics(df):

    print("\n========== STATISTICAL SUMMARY ==========")

    print(df.describe())


# ----------------------------------------
# STEP 8: Inspect target column distribution
# ----------------------------------------

def inspect_outcome_distribution(df):

    print("\n========== OUTCOME DISTRIBUTION ==========")

    if "Outcome" not in df.columns:
        print("Outcome column not found!")
        return

    print("\nPatient counts:")
    print(df["Outcome"].value_counts().sort_index())

    print("\nPatient percentages:")
    print(
        df["Outcome"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )


# ----------------------------------------
# MAIN FUNCTION
# ----------------------------------------

def main():

    try:

        df = load_dataset()

        inspect_basic_information(df)

        inspect_missing_values(df)

        inspect_zero_values(df)

        inspect_duplicates(df)

        inspect_statistics(df)

        inspect_outcome_distribution(df)

        print("\n========== INSPECTION COMPLETED ==========")

    except FileNotFoundError as error:

        print("\nERROR:", error)

    except Exception as error:

        print("\nUnexpected error:", error)


if __name__ == "__main__":
    main()
