
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ----------------------------------
# SETTINGS
# ----------------------------------

OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(exist_ok=True)


# ----------------------------------
# LOAD RESULTS
# ----------------------------------

def load_results():

    validation_file = OUTPUT_DIR / "capacity_comparison.csv"
    final_test_file = OUTPUT_DIR / "final_test_results.csv"

    if not validation_file.exists():
        raise FileNotFoundError(
            "Validation results not found. "
            "Run capacity_evaluation.py first."
        )

    if not final_test_file.exists():
        raise FileNotFoundError(
            "Final test results not found. "
            "Run select_winning_model.py first."
        )

    validation_results = pd.read_csv(validation_file)
    final_results = pd.read_csv(final_test_file)

    if validation_results.empty or final_results.empty:
        raise ValueError("One or more result files are empty.")

    if len(final_results) != 1:
        raise ValueError(
            "Expected exactly one selected model in final test results."
        )

    return validation_results, final_results


# ----------------------------------
# GENERATE MODEL COMPARISON CHART
# ----------------------------------

def generate_comparison_chart(validation_results):

    models = validation_results["Model"]

    x = range(len(models))
    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
        [i - width / 2 for i in x],
        validation_results["Precision"],
        width,
        label="Precision"
    )

    plt.bar(
        [i + width / 2 for i in x],
        validation_results["Recall"],
        width,
        label="Recall"
    )

    plt.xticks(list(x), models)
    plt.ylabel("Score")
    plt.title("Model Comparison on Validation Data")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "model_performance_comparison.png",
        dpi=300
    )

    plt.close()


# ----------------------------------
# GENERATE FINAL CAPACITY MATRIX
# ----------------------------------

def generate_capacity_matrix(winner):

    tp = int(winner["True_Positives"])
    fp = int(winner["False_Positives"])
    fn = int(winner["False_Negatives"])

    total = int(winner["Total_Patients"])

    tn = total - tp - fp - fn

    if min(tp, fp, fn, tn) < 0:
        raise ValueError(
            "Invalid screening results: negative matrix count."
        )

    confusion = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual Non-Diabetic", "Actual Diabetic"],
        columns=["Not Selected", "Selected for Testing"]
    )

    plt.figure(figsize=(7, 5))

    plt.imshow(confusion.values, cmap="Blues")
    plt.colorbar(label="Number of patients")

    plt.xticks([0, 1], confusion.columns)
    plt.yticks([0, 1], confusion.index)

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(confusion.iloc[i, j]),
                ha="center",
                va="center",
                fontsize=14
            )

    plt.title(
        f"{winner['Model']} - Final Test Capacity Selection"
    )

    plt.xlabel("Screening selection")
    plt.ylabel("Actual outcome")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "winning_model_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    return tp, fp, tn, fn


# ----------------------------------
# GENERATE TEXT REPORT
# ----------------------------------

def generate_text_report(
    validation_results,
    winner,
    tp,
    fp,
    tn,
    fn
):

    validation_ranked = validation_results.sort_values(
        by=["Recall", "Precision", "False_Negatives"],
        ascending=[False, False, True]
    )

    report = f"""
DIABETES RISK PREDICTION PROJECT
================================

1. PROJECT OBJECTIVE

Build a machine learning screening-support prototype
that ranks patients according to predicted diabetes risk.

The clinic can test a maximum of 30% of its daily patients.

2. DATASET AND EVALUATION

Final test patients: {int(winner['Total_Patients'])}
Maximum testing capacity: {int(winner['Testing_Capacity'])}
Selected patients: {int(winner['Selected_Patients'])}

3. VALIDATION MODEL COMPARISON

{validation_ranked.to_string(index=False)}

Models were compared using validation data.

Business priorities:
1. Maximize recall.
2. Maximize precision.
3. Minimize false negatives.

4. SELECTED MODEL

Selected model: {winner['Model']}

Selection was based on validation results.

5. FINAL TEST CAPACITY RESULTS

Precision: {winner['Precision']:.4f}
Recall: {winner['Recall']:.4f}

True positives: {tp}
False positives: {fp}
True negatives: {tn}
False negatives: {fn}

6. BUSINESS INTERPRETATION

The selected model ranked patients by predicted diabetes risk.

It selected {int(winner['Selected_Patients'])} out of
{int(winner['Total_Patients'])} test patients for screening.

Among the selected patients:
- {tp} were actual diabetic patients.
- {fp} were actual non-diabetic patients.

Among the patients not selected:
- {fn} were actual diabetic patients.
- {tn} were actual non-diabetic patients.

7. IMPORTANT METHODOLOGICAL NOTE

The model was selected using validation results and then
refitted on the combined training and validation data.

The final test partition was used for the final capacity
evaluation. Earlier baseline and threshold-tuning experiments
also evaluated this same partition, so this result should not
be described as a completely independent external validation.

A fresh holdout dataset or nested cross-validation would
provide a stronger estimate of generalization performance.

8. CONCLUSION

The selected model is the best candidate according to the
specified validation-based selection process.

Its final test performance is reported under a strict
30% patient-selection capacity.

Further independent validation and improvement are required
before any real-world clinical use.

9. MEDICAL DISCLAIMER

This is an educational machine learning project.

It is not a medically validated diagnostic system.
It must not replace professional medical advice,
clinical testing, or clinician decisions.

All real-world screening decisions require appropriate
clinical validation and qualified medical supervision.
"""

    report_path = OUTPUT_DIR / "final_project_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)


# ----------------------------------
# MAIN FUNCTION
# ----------------------------------

def main():

    validation_results, final_results = load_results()

    winner = final_results.iloc[0]

    generate_comparison_chart(validation_results)

    tp, fp, tn, fn = generate_capacity_matrix(winner)

    generate_text_report(
        validation_results,
        winner,
        tp,
        fp,
        tn,
        fn
    )

    print("\n========== FINAL REPORT GENERATED ==========")

    print("Selected model:", winner["Model"])
    print("Final test precision:", round(winner["Precision"], 4))
    print("Final test recall:", round(winner["Recall"], 4))

    print("\nFiles created:")
    print("outputs/model_performance_comparison.png")
    print("outputs/winning_model_confusion_matrix.png")
    print("outputs/final_project_report.txt")


if __name__ == "__main__":

    try:
        main()

    except Exception as error:
        print("\nERROR:", error)
        raise
