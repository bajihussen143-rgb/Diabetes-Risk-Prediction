
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

TESTING_CAPACITY = 0.30
RANDOM_STATE = 42


# ----------------------------------
# LOAD DATASET
# ----------------------------------

def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Cleaned dataset not found!"
        )

    return pd.read_csv(DATASET_PATH)


# ----------------------------------
# PREPARE DATA
# ----------------------------------

def prepare_data(df):

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    return X, y


# ----------------------------------
# SPLIT DATA
# ----------------------------------

def split_data(X, y):

    # Keep 20% completely untouched for final testing
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Divide the remaining 80% into training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )

    return X_train, X_val, y_train, y_val


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
# EVALUATE CAPACITY
# ----------------------------------

def evaluate_capacity(
    name,
    model,
    X_train,
    X_val,
    y_train,
    y_val
):

    print(f"\n========== {name} ==========")

    # Train using training data only
    model.fit(X_train, y_train)

    # Predict diabetes probabilities on validation data
    probabilities = model.predict_proba(X_val)[:, 1]

    # Calculate maximum number of patients to select
    capacity = math.floor(
        len(X_val) * TESTING_CAPACITY
    )

    # Rank patients from highest to lowest risk
    ranked_indices = probabilities.argsort()[::-1]

    # Select only the highest-risk patients
    selected_indices = ranked_indices[:capacity]

    selected = [False] * len(X_val)

    for index in selected_indices:
        selected[index] = True

    actual = y_val.to_numpy()

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
        actual,
        selected,
        zero_division=0
    )

    recall = recall_score(
        actual,
        selected,
        zero_division=0
    )

    print("Validation patients:", len(X_val))
    print("Testing capacity:", capacity)
    print("Selected patients:", sum(selected))
    print("True positives:", true_positives)
    print("False positives:", false_positives)
    print("False negatives:", false_negatives)
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))

    # Save validation patient rankings
    patient_results = pd.DataFrame({

        "Original_Row_Index": X_val.index,
        "Predicted_Risk": probabilities,
        "Actual_Outcome": actual,
        "Risk_Rank": ranked_indices.argsort() + 1,
        "Selected_For_Testing": selected

    }, index=X_val.index)

    patient_results["Model"] = name
    patient_results["Evaluation_Set"] = "Validation"

    patient_results = patient_results.sort_values(
        "Risk_Rank"
    )

    summary = {

        "Model": name,
        "Evaluation_Set": "Validation",
        "Total_Patients": len(X_val),
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

    try:

        OUTPUT_DIR.mkdir(exist_ok=True)

        df = load_dataset()

        X, y = prepare_data(df)

        X_train, X_val, y_train, y_val = split_data(X, y)

        models = create_models()

        summary_results = []
        patient_results = []

        for name, model in models.items():

            summary, patients = evaluate_capacity(
                name,
                model,
                X_train,
                X_val,
                y_train,
                y_val
            )

            summary_results.append(summary)
            patient_results.append(patients)

        # Save validation comparison
        summary_df = pd.DataFrame(summary_results)

        summary_df = summary_df.sort_values(
            by=["Recall", "Precision", "False_Negatives"],
            ascending=[False, False, True]
        )

        summary_df.to_csv(
            OUTPUT_DIR / "capacity_comparison.csv",
            index=False
        )

        # Save validation patient rankings
        all_patients = pd.concat(
            patient_results,
            ignore_index=True
        )

        all_patients.to_csv(
            OUTPUT_DIR / "patient_risk_rankings.csv",
            index=False
        )

        print("\n========== VALIDATION CAPACITY COMPARISON ==========")

        print(
            summary_df[
                ["Model", "Total_Patients", "Testing_Capacity",
                 "Precision", "Recall", "False_Negatives"]
            ].round(4).to_string(index=False)
        )

        print("\nValidation results saved successfully!")
        print("outputs/capacity_comparison.csv")
        print("outputs/patient_risk_rankings.csv")

        print("\nThe final test dataset has not been used.")

    except Exception as error:

        print("\nERROR:", error)


if __name__ == "__main__":
    main()
