
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt

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
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ----------------------------------
# File paths
# ----------------------------------

DATASET_PATH = Path("data/diabetes_cleaned.csv")
OUTPUT_DIR = Path("outputs")
CONFUSION_DIR = OUTPUT_DIR / "confusion_matrices"


# ----------------------------------
# Load dataset
# ----------------------------------

def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Cleaned dataset not found!"
        )

    return pd.read_csv(DATASET_PATH)


# ----------------------------------
# Prepare and split data
# ----------------------------------

def prepare_data(df):

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


# ----------------------------------
# Create models
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
# Evaluate one model
# ----------------------------------

def evaluate_one_model(name, model, X_train, X_test, y_train, y_test):

    print(f"\nEvaluating {name}...")

    # Train the model
    model.fit(X_train, y_train)

    # Predict test outcomes
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    print("\nAccuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))

    print("\nConfusion Matrix:")
    print(cm)

    # Save confusion matrix image
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Diabetic", "Diabetic"]
    )

    display.plot(cmap="Blues", values_format="d")

    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()

    safe_name = name.lower().replace(" ", "_")

    plt.savefig(
        CONFUSION_DIR / f"{safe_name}_confusion_matrix.png"
    )

    plt.close()

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "True_Negative": int(cm[0, 0]),
        "False_Positive": int(cm[0, 1]),
        "False_Negative": int(cm[1, 0]),
        "True_Positive": int(cm[1, 1])
    }


# ----------------------------------
# Evaluate all models
# ----------------------------------

def evaluate_all_models(models, X_train, X_test, y_train, y_test):

    results = []

    for name, model in models.items():

        result = evaluate_one_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append(result)

    return pd.DataFrame(results)


# ----------------------------------
# Main function
# ----------------------------------

def main():

    try:

        OUTPUT_DIR.mkdir(exist_ok=True)
        CONFUSION_DIR.mkdir(exist_ok=True)

        df = load_dataset()

        X_train, X_test, y_train, y_test = prepare_data(df)

        models = create_models()

        results_df = evaluate_all_models(
            models,
            X_train,
            X_test,
            y_train,
            y_test
        )

        # Sort by Recall for initial comparison
        results_df = results_df.sort_values(
            by="Recall",
            ascending=False
        )

        print("\n========== MODEL COMPARISON ==========")

        print(
            results_df[
                ["Model", "Accuracy", "Precision", "Recall", "F1_Score"]
            ].round(4).to_string(index=False)
        )

        # Save results
        results_path = OUTPUT_DIR / "evaluation_results.csv"

        results_df.to_csv(
            results_path,
            index=False
        )

        print("\nEvaluation results saved to:", results_path)

        print(
            "\nConfusion matrices saved in:",
            CONFUSION_DIR
        )


    except Exception as error:

        print("\nERROR:", error)


if __name__ == "__main__":
    main()
