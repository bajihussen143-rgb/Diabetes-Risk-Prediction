
import pandas as pd
import pytest

from predict_diabetes import (
    FEATURES,
    get_patient_data,
    load_model,
    predict_risk,
)


def test_saved_model_loads_successfully():
    model = load_model()

    assert model is not None
    assert hasattr(model, "predict_proba")


def test_prediction_returns_valid_percentage():
    model = load_model()

    patient = pd.DataFrame(
        [[2, 120, 70, 25, 80, 28, 0.5, 30]],
        columns=FEATURES,
    )

    risk = predict_risk(model, patient)

    assert 0 <= risk <= 100


def test_patient_data_has_correct_features(monkeypatch):
    values = iter([
        "2", "120", "70", "25",
        "80", "28", "0.5", "30"
    ])

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(values)
    )

    patient = get_patient_data()

    assert list(patient.columns) == FEATURES
    assert patient.shape == (1, 8)


def test_invalid_values_are_rejected(monkeypatch):
    values = iter([
        "2.5", "2",
        "0", "120",
        "-5", "70",
        "25", "80", "28", "0.5", "30"
    ])

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(values)
    )

    patient = get_patient_data()

    assert patient.shape == (1, 8)
    assert patient.iloc[0]["Pregnancies"] == 2
    assert patient.iloc[0]["Glucose"] == 120
    assert patient.iloc[0]["BloodPressure"] == 70
