
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split


# ----------------------------------
# File paths
# ----------------------------------

INPUT_PATH = Path("data/diabetes_cleaned.csv")
OUTPUT_DIR = Path("outputs")


# ----------------------------------
# Load cleaned dataset
# ----------------------------------

def load_dataset():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Cleaned dataset not found. Run clean_dataset.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    print("Cleaned dataset loaded successfully!")

    return df


# ----------------------------------
# Separate features and target
# ----------------------------------

def prepare_features_target(df):

    if "Outcome" not in df.columns:
        raise ValueError("Outcome column is missing!")

    X = df.drop(columns=["Outcome"])

    y = df["Outcome"]

    print("\nFeature columns:")
    print(X.columns.tolist())

    print("\nTarget column:")
    print(y.name)

    return X, y


# ----------------------------------
# Stratified train-test split
# ----------------------------------

def split_dataset(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\n========== DATASET SPLIT ==========")

    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)

    print("Training target:", y_train.shape)
    print("Testing target:", y_test.shape)

    return X_train, X_test, y_train, y_test


# ----------------------------------
# Check class distribution
# ----------------------------------

def check_distribution(y, name):

    print(f"\n{name} Outcome distribution:")

    print(y.value_counts().sort_index())

    print("\nPercentage distribution:")

    print(
        (y.value_counts(normalize=True).sort_index() * 100).round(2)
    )


# ----------------------------------
# Main function
# ----------------------------------

def main():

    try:

        df = load_dataset()

        X, y = prepare_features_target(df)

        X_train, X_test, y_train, y_test = split_dataset(X, y)

        check_distribution(y, "Original dataset")

        check_distribution(y_train, "Training dataset")

        check_distribution(y_test, "Testing dataset")

        print("\n========== STEP 4 COMPLETED ==========")

    except Exception as error:

        print("\nERROR:", error)


if __name__ == "__main__":
    main()
