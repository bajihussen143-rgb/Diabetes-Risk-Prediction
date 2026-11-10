
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/diabetes_risk_model.joblib")

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

POSITIVE_REQUIRED_FIELDS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. "
            "Run select_winning_model.py first."
        )

    return joblib.load(MODEL_PATH)


def get_patient_data():
    print("\n========== PATIENT INFORMATION ==========")

    patient = {}

    for feature in FEATURES:
        while True:
            try:
                value = float(input(f"Enter {feature}: "))

                if value < 0:
                    print("Value cannot be negative.")
                    continue

                if feature == "Pregnancies" and not value.is_integer():
                    print("Pregnancies must be a whole number.")
                    continue

                if feature in POSITIVE_REQUIRED_FIELDS and value == 0:
                    print(
                        f"{feature} cannot be zero. "
                        "Enter a valid measurement."
                    )
                    continue

                patient[feature] = value
                break

            except ValueError:
                print("Please enter a valid number.")

    return pd.DataFrame([patient], columns=FEATURES)


def predict_risk(model, patient_data):
    probability = model.predict_proba(patient_data)[0][1]
    return probability * 100


def display_result(risk_percentage):
    print("\n========== PREDICTION RESULT ==========")

    print(
        f"Estimated model risk score: "
        f"{risk_percentage:.2f}%"
    )

    print("\nInterpretation:")
    print(
        "This is a model-estimated probability, "
        "not a medically validated risk estimate."
    )
    print(
        "It is not a diagnosis and should not replace "
        "clinical assessment or confirmatory testing."
    )
    print(
        "\nA qualified healthcare professional "
        "should interpret the result."
    )


def main():
    print("DIABETES RISK PREDICTION SYSTEM")

    model = load_model()

    patient_data = get_patient_data()

    risk_percentage = predict_risk(model, patient_data)

    display_result(risk_percentage)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nERROR:", error)
        raise
