
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ----------------------------------
# File paths
# ----------------------------------

INPUT_PATH = Path("data/diabetes_cleaned.csv")


# ----------------------------------
# Load cleaned dataset
# ----------------------------------

def load_dataset():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Cleaned dataset not found!"
        )

    df = pd.read_csv(INPUT_PATH)

    print("Dataset loaded successfully!")

    return df


# ----------------------------------
# Separate features and target
# ----------------------------------

def prepare_features_target(df):

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    return X, y


# ----------------------------------
# Split dataset
# ----------------------------------

def split_dataset(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


# ----------------------------------
# Handle missing values
# ----------------------------------

def impute_missing_values(X_train, X_test):

    imputer = SimpleImputer(strategy="median")

    # Learn medians from training data only
    X_train_imputed = imputer.fit_transform(X_train)

    # Apply the same medians to testing data
    X_test_imputed = imputer.transform(X_test)

    print("\nMissing values handled using median imputation.")

    return X_train_imputed, X_test_imputed, imputer


# ----------------------------------
# Apply feature scaling
# ----------------------------------

def scale_features(X_train, X_test):

    scaler = StandardScaler()

    # Learn scaling values from training data only
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply the same scaling values to testing data
    X_test_scaled = scaler.transform(X_test)

    print("Feature scaling completed using StandardScaler.")

    return X_train_scaled, X_test_scaled, scaler


# ----------------------------------
# Main function
# ----------------------------------

def main():

    try:

        df = load_dataset()

        X, y = prepare_features_target(df)

        X_train, X_test, y_train, y_test = split_dataset(X, y)

        print("\nOriginal training shape:", X_train.shape)
        print("Original testing shape:", X_test.shape)

        # Step 1: Imputation
        X_train_imputed, X_test_imputed, imputer = (
            impute_missing_values(X_train, X_test)
        )

        # Step 2: Scaling
        X_train_scaled, X_test_scaled, scaler = (
            scale_features(X_train_imputed, X_test_imputed)
        )

        print("\n========== PREPROCESSING RESULTS ==========")

        print("Training data shape:", X_train_scaled.shape)
        print("Testing data shape:", X_test_scaled.shape)

        print(
            "\nRemaining missing values in training:",
            pd.isna(X_train_scaled).sum()
        )

        print(
            "Remaining missing values in testing:",
            pd.isna(X_test_scaled).sum()
        )

        print("\nFirst 5 scaled training records:")

        print(X_train_scaled[:5])

        print("\n========== STEP 5 COMPLETED ==========")

    except Exception as error:

        print("\nERROR:", error)


if __name__ == "__main__":
    main()
