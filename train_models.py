
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# ----------------------------------
# Dataset location
# ----------------------------------

DATASET_PATH = Path("data/diabetes_cleaned.csv")


# ----------------------------------
# Load dataset
# ----------------------------------

def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Cleaned dataset not found. Run clean_dataset.py first."
        )

    df = pd.read_csv(DATASET_PATH)

    print("Dataset loaded successfully!")

    return df


# ----------------------------------
# Prepare features and target
# ----------------------------------

def prepare_data(df):

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    return X, y


# ----------------------------------
# Split dataset
# ----------------------------------

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    return X_train, X_test, y_train, y_test


# ----------------------------------
# Create machine learning pipelines
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
            ("model", LogisticRegression(max_iter=1000, random_state=42))
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
# Train all models
# ----------------------------------

def train_models(models, X_train, y_train):

    trained_models = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        trained_models[name] = model

        print(f"{name} training completed!")

    return trained_models


# ----------------------------------
# Main function
# ----------------------------------

def main():

    try:

        df = load_dataset()

        X, y = prepare_data(df)

        X_train, X_test, y_train, y_test = split_data(X, y)

        models = create_models()

        trained_models = train_models(
            models,
            X_train,
            y_train
        )

        print("\n========== TRAINING SUMMARY ==========")

        for name in trained_models:
            print(f"{name}: Successfully trained")

        print("\nTraining records:", X_train.shape)
        print("Testing records:", X_test.shape)


    except Exception as error:

        print("\nERROR:", error)


if __name__ == "__main__":
    main()
