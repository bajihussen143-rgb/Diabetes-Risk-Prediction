# Diabetes Risk Prediction System

A machine-learning project that identifies patients who may be at risk of diabetes and prioritizes them for screening when a healthcare clinic has limited testing capacity.

## 1. Project Overview

Healthcare clinics may not have enough resources to test every patient visiting each day. This project explores how machine learning can help prioritize patients who are more likely to have diabetes.

The system trains and compares three classification models:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree

Instead of simply classifying every patient, the project also ranks patients by predicted diabetes risk and selects the highest-risk 30% for screening.

**Important:** This is an educational screening-support prototype. It is not a medically validated diagnostic system and must not be used to make real clinical decisions.

## 2. Business Problem

Imagine a rural healthcare clinic that can test only 30% of its daily patients.

The clinic needs a system that can:

- Identify patients with potentially higher diabetes risk.
- Prioritize patients for limited screening capacity.
- Reduce missed diabetic cases.
- Compare models using recall and precision, rather than accuracy alone.

The objective is to catch as many actual diabetic patients as possible within the available testing capacity.

## 3. Dataset

**Dataset:** Pima Indians Diabetes Database

**Source:** [Kaggle Dataset](https://www.kaggle.com/code/mragpavank/pima-indians-diabetesdatabase/input)

The dataset contains 768 patient records and 8 input features.

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | 2-hour serum insulin |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Patient age |
| Outcome | Diabetes result (1 = diabetic, 0 = non-diabetic) |

## 4. Data Preprocessing

The project includes:

- Loading and inspecting the original CSV dataset.
- Identifying hidden missing values represented by zero.
- Replacing invalid zero values with missing values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI.
- Applying median imputation.
- Applying StandardScaler for feature scaling.
- Using stratified train-test splitting.
- Preventing preprocessing leakage by fitting preprocessing steps on training data only.

Zero values in valid fields such as Pregnancies and Outcome are preserved.

## 5. Machine Learning Models

Three models are trained and compared:

1. K-Nearest Neighbors
2. Logistic Regression
3. Decision Tree

The project evaluates:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

It also includes probability threshold tuning using validation data.

## 6. Limited Testing Capacity

The clinic can test only 30% of patients.

The system:

1. Predicts diabetes probabilities.
2. Ranks patients from highest to lowest predicted risk.
3. Selects the highest-risk patients within the available capacity.
4. Evaluates how many actual diabetic patients were selected.

The maximum testing capacity is calculated using:

`floor(total_patients * 0.30)`

This ensures the system never exceeds the clinic's available testing capacity.

## 7. Model Selection and Results

Models are compared using validation data. Recall is prioritized, followed by precision and fewer false negatives.

### Validation results

| Model | Precision | Recall | False Negatives |
|---|---:|---:|---:|
| KNN | 75.00% | 62.79% | 16 |
| Logistic Regression | 75.00% | 62.79% | 16 |
| Decision Tree | 66.67% | 55.81% | 19 |

KNN and Logistic Regression achieved identical validation results. KNN was selected using the original model order as the tie-breaker.

### Final test results

| Metric | Result |
|---|---:|
| Selected model | KNN |
| Total test patients | 154 |
| Maximum testing capacity | 46 |
| Selected patients | 46 |
| True positives | 30 |
| False positives | 16 |
| False negatives | 24 |
| Precision | 65.22% |
| Recall | 55.56% |

These results show how the selected model performed when prioritizing patients within the 30% screening capacity.

**Evaluation limitation:** Earlier baseline evaluation and threshold-tuning experiments also used the same test partition. Therefore, these final results should not be interpreted as an entirely untouched independent holdout evaluation. A fresh holdout dataset or nested cross-validation would provide stronger evidence of generalization.

## 8. Project Structure

```text
Diabetes_Risk_Prediction/
│
├── data/
│   ├── diabetes.csv
│   └── diabetes_cleaned.csv
│
├── outputs/
│   ├── evaluation_results.csv
│   ├── threshold_validation_results.csv
│   ├── threshold_test_results.csv
│   ├── capacity_comparison.csv
│   ├── patient_risk_rankings.csv
│   ├── winning_model.txt
│   ├── final_test_results.csv
│   ├── final_test_patient_rankings.csv
│   ├── final_project_report.txt
│   ├── model_performance_comparison.png
│   └── winning_model_confusion_matrix.png
│
├── models/
├── tests/
│   └── test_project.py
│
├── inspect_dataset.py
├── clean_dataset.py
├── prepare_dataset.py
├── preprocess_dataset.py
├── train_models.py
├── evaluate_models.py
├── threshold_tuning.py
├── capacity_evaluation.py
├── select_winning_model.py
├── generate_report.py
│
├── requirements.txt
└── README.md
```

## 9. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Pytest
- VS Code

## 10. Installation and Execution

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Diabetes_Risk_Prediction
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

### Step 3: Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Add the dataset

Place the downloaded `diabetes.csv` file inside the `data/` directory.

### Step 6: Run the project scripts

Run these scripts from the project root in the following order:

```bash
python inspect_dataset.py
python clean_dataset.py
python prepare_dataset.py
python preprocess_dataset.py
python train_models.py
python evaluate_models.py
python threshold_tuning.py
python capacity_evaluation.py
python select_winning_model.py
python generate_report.py
```

### Step 7: Run automated tests

```bash
python -m pytest -v
```

## 11. Automated Testing

The project includes automated tests for:

- Dataset structure and target values.
- Hidden missing-value handling.
- Validation screening capacity.
- Validation patient rankings.
- Final-test screening capacity.
- Final-test patient rankings.
- Validation-based model selection.

All 7 project tests passed during development.

## 12. Future Improvements

- Validate the model using a new independent dataset.
- Explore additional machine-learning algorithms.
- Improve recall while respecting screening capacity.
- Add model persistence and a prediction API.
- Develop a web dashboard for healthcare screening workflows.
- Evaluate fairness and performance across different patient groups.
- Obtain clinical validation before any real-world use.

## 13. Disclaimer

This project is developed for educational and machine-learning learning purposes only. Its predictions are not medical diagnoses or treatment recommendations. Real-world deployment would require independent clinical validation, appropriate confirmatory testing, privacy safeguards, and qualified healthcare professional oversight.
