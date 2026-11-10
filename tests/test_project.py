
import math
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MISSING_VALUE_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

EXPECTED_MODELS = {
    "KNN",
    "Logistic Regression",
    "Decision Tree",
}


def test_cleaned_dataset_has_no_hidden_zero_values():
    df = pd.read_csv(DATA_DIR / "diabetes_cleaned.csv")

    for column in MISSING_VALUE_COLUMNS:
        assert not (df[column] == 0).any(), (
            f"Hidden zero values remain in {column}"
        )


def test_dataset_has_expected_columns_and_target():
    df = pd.read_csv(DATA_DIR / "diabetes_cleaned.csv")

    expected_columns = {
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Outcome",
    }

    assert set(df.columns) == expected_columns
    assert set(df["Outcome"].dropna().unique()).issubset({0, 1})


def test_validation_capacity_never_exceeds_30_percent():
    results = pd.read_csv(
        OUTPUT_DIR / "capacity_comparison.csv"
    )

    assert set(results["Model"]) == EXPECTED_MODELS
    assert set(results["Evaluation_Set"]) == {"Validation"}

    for _, row in results.iterrows():
        maximum_capacity = math.floor(
            row["Total_Patients"] * 0.30
        )

        assert row["Testing_Capacity"] == maximum_capacity
        assert row["Selected_Patients"] <= maximum_capacity


def test_validation_patient_rankings_respect_capacity():
    rankings = pd.read_csv(
        OUTPUT_DIR / "patient_risk_rankings.csv"
    )

    assert set(rankings["Model"]) == EXPECTED_MODELS

    for model_name, group in rankings.groupby("Model"):
        total_patients = len(group)
        maximum_capacity = math.floor(total_patients * 0.30)

        selected = group[
            group["Selected_For_Testing"].astype(str).str.lower()
            .isin(["true", "1"])
        ]

        assert len(selected) == maximum_capacity, (
            f"{model_name} does not use the full available capacity"
        )

        assert group["Risk_Rank"].nunique() == total_patients

        expected_selected = group.nsmallest(
            maximum_capacity, "Risk_Rank"
        )

        assert set(selected.index) == set(expected_selected.index)


def test_final_test_capacity_never_exceeds_30_percent():
    results = pd.read_csv(
        OUTPUT_DIR / "final_test_results.csv"
    )

    assert len(results) == 1
    row = results.iloc[0]

    assert row["Evaluation_Set"] == "Final Test"

    maximum_capacity = math.floor(
        row["Total_Patients"] * 0.30
    )

    assert row["Testing_Capacity"] == maximum_capacity
    assert row["Selected_Patients"] <= maximum_capacity


def test_final_test_patient_rankings_respect_capacity():
    rankings = pd.read_csv(
        OUTPUT_DIR / "final_test_patient_rankings.csv"
    )

    assert rankings["Model"].nunique() == 1

    group = rankings
    total_patients = len(group)
    maximum_capacity = math.floor(total_patients * 0.30)

    selected = group[
        group["Selected_For_Testing"].astype(str).str.lower()
        .isin(["true", "1"])
    ]

    assert len(selected) == maximum_capacity
    assert group["Risk_Rank"].nunique() == total_patients

    expected_selected = group.nsmallest(
        maximum_capacity, "Risk_Rank"
    )

    assert set(selected.index) == set(expected_selected.index)


def test_winning_model_is_selected_from_validation_results():
    report_path = OUTPUT_DIR / "winning_model.txt"
    results_path = OUTPUT_DIR / "capacity_comparison.csv"

    assert report_path.exists()
    assert results_path.exists()

    report = report_path.read_text(encoding="utf-8")
    results = pd.read_csv(results_path)

    assert "Model: KNN" in report
    assert set(results["Evaluation_Set"]) == {"Validation"}

    sorted_results = results.sort_values(
        by=["Recall", "Precision", "False_Negatives"],
        ascending=[False, False, True],
        kind="stable",
    )

    expected_winner = sorted_results.iloc[0]["Model"]

    assert f"Model: {expected_winner}" in report
