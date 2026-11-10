
import math
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import precision_score, recall_score


# ----------------------------------
# SETTINGS
# ----------------------------------

DATASET_PATH = Path("data/diabetes_cleaned.csv")
OUTPUT_DIR = Path("outputs")

CAPACITY = 0.30
RANDOM_STATE = 42


# ----------------------------------
# LOAD VALIDATION RESULTS
# ----------------------------------

def load_validation_results():

    path = OUTPUT_DIR / "capacity_comparison.csv"

    if not path.exists():
        raise FileNotFoundError(
            "Validation results not found. "
            "Run capacity_evaluation.py first."
        )

    results = pd.read_csv(path)

    if results.empty:
        raise ValueError("Validation results are empty.")

    return results


# ----------------------------------
# SELECT WINNING MODEL
# ----------------------------------

def select_winner(results):

    ranked = results.sort_values(
        by=["Recall", "Precision", "False_Negatives"],
        ascending=[False, False, True],
        kind="stable"
    ).reset_index(drop=True)

    winner = ranked.iloc[0]

    print("\n========== VALIDATION MODEL COMPARISON ==========")

    print(
        ranked[
            ["Model", "Recall", "Precision", "False_Negatives"]
        ].to_string(index=False)
    )

    print("\n========== SELECTED MODEL ==========")
    print("Model:", winner["Model"])
    print("Validation recall:", round(winner["Recall"], 4))
    print("Validation precision:", round(winner["Precision"], 4))
    print("Validation false negatives:", int(winner["False_Negatives"]))

    print(
        "\nSelection is based on validation results only. "
        "Exact ties are resolved by the original model order."
    )

    return winner["Model"]


# ----------------------------------
# LOAD AND SPLIT DATA
# ----------------------------------

def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError("Cleaned dataset not found.")

    return pd.read_csv(DATASET_PATH)


def prepare_data(df):

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    return X, y


def split_data(X, y):

    # Reserve 20% for final evaluation
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Validation split is used for model selection.
    # The selected model will later be retrained on all 80%.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )

    return X_train_full, X_test, y_train_full, y_test


# ----------------------------------
# CREATE MODELS
# ----------------------------------

def create_models():

    return {

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
                random_state=RANDOM_STATE
            ))
        ]),

        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(
                max_depth=5,
                random_state=RANDOM_STATE
            ))
        ])

    }


# ----------------------------------
# FINAL CAPACITY EVALUATION
# ----------------------------------

def evaluate_final_model(
    name,
    model,
    X_train_full,
    X_test,
    y_train_full,
    y_test
):

    # Retrain using the complete 80% training partition
    model.fit(X_train_full, y_train_full)

    # Predict risk probabilities for test patients
    probabilities = model.predict_proba(X_test)[:, 1]

    # Strictly enforce the 30% capacity limit
    capacity = math.floor(len(X_test) * CAPACITY)

    # Rank patients from highest to lowest predicted risk
    ranked_indices = probabilities.argsort()[::-1]

    selected_indices = ranked_indices[:capacity]

    selected = [False] * len(X_test)

    for index in selected_indices:
        selected[index] = True

    actual = y_test.to_numpy()

    # Calculate screening outcomes
    true_positives = sum(
        selected[i] and actual[i] == 1
        for i in range(len(actual))
    )

    false_positives = sum(
        selected[i] and actual[i] == 0
        for i in range(len(actual))
    )

    false_negatives = sum(
        not selected[i] and actual[i] == 1
        for i in range(len(actual))
    )

    precision = precision_score(
        actual, selected, zero_division=0
    )

    recall = recall_score(
        actual, selected, zero_division=0
    )

    # Save final test patient rankings
    patient_results = pd.DataFrame({

        "Original_Row_Index": X_test.index,
        "Predicted_Risk": probabilities,
        "Actual_Outcome": actual,
        "Risk_Rank": ranked_indices.argsort() + 1,
        "Selected_For_Testing": selected

    }, index=X_test.index)

    patient_results["Model"] = name
    patient_results["Evaluation_Set"] = "Final Test"

    patient_results = patient_results.sort_values("Risk_Rank")

    summary = {

        "Model": name,
        "Evaluation_Set": "Final Test",
        "Total_Patients": len(X_test),
        "Testing_Capacity": capacity,
        "Selected_Patients": sum(selected),
        "True_Positives": true_positives,
        "False_Positives": false_positives,
        "False_Negatives": false_negatives,
        "Precision": precision,
        "Recall": recall

    }

    return summary, patient_results


# ----------------------------------
# MAIN FUNCTION
# ----------------------------------

def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Step 1: Select model using validation results
    validation_results = load_validation_results()
    winning_model_name = select_winner(validation_results)

    # Step 2: Load dataset and reproduce the same split
    df = load_dataset()
    X, y = prepare_data(df)

    X_train_full, X_test, y_train_full, y_test = split_data(X, y)

    # Step 3: Create and retrieve the selected model
    models = create_models()

    if winning_model_name not in models:
        raise ValueError(
            f"Unknown model in validation results: {winning_model_name}"
        )

    winning_model = models[winning_model_name]

    # Step 4: Evaluate the selected model on the final test partition
    final_summary, patient_results = evaluate_final_model(
        winning_model_name,
        winning_model,
        X_train_full,
        X_test,
        y_train_full,
        y_test
    )

    # Save final test metrics
    pd.DataFrame([final_summary]).to_csv(
        OUTPUT_DIR / "final_test_results.csv",
        index=False
    )

    # Save final test patient rankings
    patient_results.to_csv(
        OUTPUT_DIR / "final_test_patient_rankings.csv",
        index=False
    )

    # Save winning model details
    with open(
        OUTPUT_DIR / "winning_model.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write("WINNING MODEL SELECTION\n")
        file.write("=======================\n")
        file.write(f"Model: {winning_model_name}\n")
        file.write("Selection basis: Validation results\n")
        file.write(f"Final test precision: {final_summary['Precision']:.4f}\n")
        file.write(f"Final test recall: {final_summary['Recall']:.4f}\n")
        file.write(
            f"Final test false negatives: "
            f"{final_summary['False_Negatives']}\n"
        )

    # Display final test results
    print("\n========== FINAL TEST EVALUATION ==========")

    for key, value in final_summary.items():
        print(f"{key}: {value}")

    print("\nFiles created:")
    print("outputs/winning_model.txt")
    print("outputs/final_test_results.csv")
    print("outputs/final_test_patient_rankings.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nERROR:", error)
        raise
