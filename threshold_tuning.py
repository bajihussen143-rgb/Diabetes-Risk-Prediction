
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score
)


# ----------------------------------
# File paths and settings
# ----------------------------------

DATASET_PATH = Path("data/diabetes_cleaned.csv")
OUTPUT_DIR = Path("outputs")

THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]


# ----------------------------------
# Load dataset
# ----------------------------------

def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError("Cleaned dataset not found!")

    return pd.read_csv(DATASET_PATH)


# ----------------------------------
# Prepare features and target
# ----------------------------------

def prepare_data(df):

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    return X, y


# ----------------------------------
# Create train, validation and test sets
# ----------------------------------

def split_data(X, y):

    # Keep 20% for final testing
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Split remaining 80% into training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=42,
        stratify=y_train_full
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ----------------------------------
# Create model pipelines
# ----------------------------------

def create_models():

    models = {

        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),

        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ]),

        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            ))
        ])

    }

    return models


# ----------------------------------
# Calculate metrics for a threshold
# ----------------------------------

def calculate_metrics(y_true, probabilities, threshold):

    predictions = (probabilities >= threshold).astype(int)

    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y_true, predictions, zero_division=0
        ),
        "F2_Score": fbeta_score(
            y_true, predictions, beta=2, zero_division=0
        )
    }


# ----------------------------------
# Tune threshold using validation data
# ----------------------------------

def tune_model(name, model, X_train, X_val, y_train, y_val):

    print(f"\n========== {name} ==========")

    # Train only on training data
    model.fit(X_train, y_train)

    # Obtain probability of diabetic class (class 1)
    probabilities = model.predict_proba(X_val)[:, 1]

    results = []

    for threshold in THRESHOLDS:

        metrics = calculate_metrics(
            y_val,
            probabilities,
            threshold
        )

        results.append(metrics)

    results_df = pd.DataFrame(results)

    print("\nValidation threshold results:")

    print(results_df.round(4).to_string(index=False))

    # Select threshold with highest F2 score
    best_row = results_df.loc[
        results_df["F2_Score"].idxmax()
    ]

    best_threshold = float(best_row["Threshold"])

    print("\nSelected threshold:", best_threshold)

    return model, best_threshold, results_df


# ----------------------------------
# Evaluate selected threshold on test data
# ----------------------------------

def evaluate_test_data(
    name,
    model,
    threshold,
    X_test,
    y_test
):

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(
        y_test,
        probabilities,
        threshold
    )

    print(f"\n{name} final test results:")
    print(pd.Series(metrics).round(4))

    return {
        "Model": name,
        **metrics
    }


# ----------------------------------
# Main function
# ----------------------------------

def main():

    try:

        OUTPUT_DIR.mkdir(exist_ok=True)

        df = load_dataset()

        X, y = prepare_data(df)

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        ) = split_data(X, y)

        print("Training records:", len(X_train))
        print("Validation records:", len(X_val))
        print("Testing records:", len(X_test))

        models = create_models()

        all_validation_results = []
        all_test_results = []

        for name, model in models.items():

            trained_model, best_threshold, validation_results = (
                tune_model(
                    name,
                    model,
                    X_train,
                    X_val,
                    y_train,
                    y_val
                )
            )

            validation_results.insert(0, "Model", name)

            all_validation_results.append(validation_results)

            test_result = evaluate_test_data(
                name,
                trained_model,
                best_threshold,
                X_test,
                y_test
            )

            all_test_results.append(test_result)

        # Save validation threshold results
        validation_df = pd.concat(
            all_validation_results,
            ignore_index=True
        )

        validation_df.to_csv(
            OUTPUT_DIR / "threshold_validation_results.csv",
            index=False
        )

        # Save final test results
        test_df = pd.DataFrame(all_test_results)

        test_df.to_csv(
            OUTPUT_DIR / "threshold_test_results.csv",
            index=False
        )

        print("\n========== FINAL TEST COMPARISON ==========")

        print(test_df.round(4).to_string(index=False))

        print("\nResults saved inside outputs folder.")


    except Exception as error:

        print("\nERROR:", error)


if __name__ == "__main__":
    main()
