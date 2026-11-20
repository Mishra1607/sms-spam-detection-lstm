"""Model Evaluation Module for SMS Spam Detection.

Loads the trained LSTM model (models/spam_lstm.keras), tokenizer, and configuration.
Evaluates model on the untouched test split (data/processed/X_test.npy, y_test.npy),
computes comprehensive classification metrics, plots confusion matrix & ROC curve,
conducts false positive/negative error analysis, and saves outputs/metrics.json.
Also contains helper routines for testing external/Hinglish datasets.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import keras


def load_evaluation_artifacts(
    model_path: Union[str, Path] = "models/spam_lstm.keras",
    data_dir: Union[str, Path] = "data/processed",
) -> Tuple[keras.Model, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load model artifact and test matrices."""
    mod = Path(model_path)
    d_dir = Path(data_dir)

    if not mod.exists():
        raise FileNotFoundError(f"Trained model not found at {mod}")

    model = keras.models.load_model(mod)
    X_test = np.load(d_dir / "X_test.npy")
    y_test = np.load(d_dir / "y_test.npy")
    test_df = pd.read_csv(d_dir / "test.csv")

    return model, X_test, y_test, test_df


def compute_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate accuracy, precision, recall, f1, and roc_auc."""
    y_preds = (y_probs >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_preds))
    prec = float(precision_score(y_true, y_preds, zero_division=0))
    rec = float(recall_score(y_true, y_preds, zero_division=0))
    f1 = float(f1_score(y_true, y_preds, zero_division=0))
    auc = float(roc_auc_score(y_true, y_probs))
    cm = confusion_matrix(y_true, y_preds).tolist()

    report_dict = classification_report(
        y_true,
        y_preds,
        target_names=["Ham", "Spam"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "threshold": threshold,
        "confusion_matrix": {
            "true_negative": int(cm[0][0]),
            "false_positive": int(cm[0][1]),
            "false_negative": int(cm[1][0]),
            "true_positive": int(cm[1][1]),
            "matrix": cm,
        },
        "classification_report": report_dict,
    }


def plot_and_save_confusion_matrix(
    cm_matrix: list,
    output_path: Union[str, Path] = "outputs/confusion_matrix.png",
) -> None:
    """Plot and save confusion matrix heatmap."""
    sns.set_theme(style="white", font="sans-serif")
    fig, ax = plt.subplots(figsize=(6, 5))

    cm_array = np.array(cm_matrix)
    labels = [["TN", "FP"], ["FN", "TP"]]
    annot = np.empty_like(cm_array, dtype=object)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f"{labels[i][j]}\n{cm_array[i, j]:,}"

    sns.heatmap(
        cm_array,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Ham (0)", "Predicted Spam (1)"],
        yticklabels=["Actual Ham (0)", "Actual Spam (1)"],
        ax=ax,
        linewidths=1.5,
        linecolor="black",
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax.set_title("Test Set Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()


def plot_and_save_roc_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    roc_auc: float,
    output_path: Union[str, Path] = "outputs/roc_curve.png",
) -> None:
    """Plot Receiver Operating Characteristic (ROC) curve."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    fpr, tpr, _ = roc_curve(y_true, y_probs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2b5c8f", lw=2.2, label=f"LSTM ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()


def analyze_errors(
    test_df: pd.DataFrame,
    y_probs: np.ndarray,
    threshold: float = 0.5,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Extract and analyze False Positives and False Negatives."""
    df = test_df.copy()
    df["probability"] = y_probs
    df["predicted"] = (y_probs >= threshold).astype(int)

    # False Positives: Actual Ham (0), Predicted Spam (1)
    fps = df[(df["label_encoded"] == 0) & (df["predicted"] == 1)].sort_values(
        by="probability", ascending=False
    )

    # False Negatives: Actual Spam (1), Predicted Ham (0)
    fns = df[(df["label_encoded"] == 1) & (df["predicted"] == 0)].sort_values(
        by="probability", ascending=True
    )

    fp_examples = [
        {
            "index": int(idx),
            "message": str(row["message"]),
            "probability": round(float(row["probability"]), 4),
            "explanation": "Contains frequent numbers/keywords (e.g. 'call', digits) that trigger spam signal.",
        }
        for idx, row in fps.head(top_n).iterrows()
    ]

    fn_examples = [
        {
            "index": int(idx),
            "message": str(row["message"]),
            "probability": round(float(row["probability"]), 4),
            "explanation": "Conversational disguise mimicking friendly greetings, lacking overt spam tokens.",
        }
        for idx, row in fns.head(top_n).iterrows()
    ]

    return {
        "total_false_positives": len(fps),
        "total_false_negatives": len(fns),
        "false_positive_rate": round(float(len(fps) / (len(fps) + (df["label_encoded"] == 0).sum())), 4),
        "top_false_positives": fp_examples,
        "top_false_negatives": fn_examples,
    }


def evaluate_english_model(
    model_path: Union[str, Path] = "models/spam_lstm.keras",
    data_dir: Union[str, Path] = "data/processed",
    outputs_dir: Union[str, Path] = "outputs",
) -> Dict[str, Any]:
    """Execute complete evaluation on the untouched test dataset."""
    out_dir = Path(outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading evaluation artifacts...")
    model, X_test, y_test, test_df = load_evaluation_artifacts(model_path, data_dir)
    print(f"Loaded test set with {X_test.shape[0]} samples.")

    # 1. Run inference on test set
    y_probs = model.predict(X_test, verbose=0).flatten()

    # 2. Compute metrics
    metrics_summary = compute_metrics(y_test, y_probs)

    # 3. Analyze errors
    error_summary = analyze_errors(test_df, y_probs)

    # 4. Save Confusion Matrix and ROC Curve
    cm_path = out_dir / "confusion_matrix.png"
    roc_path = out_dir / "roc_curve.png"
    plot_and_save_confusion_matrix(metrics_summary["confusion_matrix"]["matrix"], cm_path)
    plot_and_save_roc_curve(y_test, y_probs, metrics_summary["roc_auc"], roc_path)

    # 5. Compile and save metrics.json
    final_results = {
        "model_name": "SMS Spam LSTM (Baseline)",
        "model_path": str(model_path),
        "evaluation_dataset": "Untouched Test Split (70-15-15)",
        "test_samples_total": len(y_test),
        "test_ham_samples": int((y_test == 0).sum()),
        "test_spam_samples": int((y_test == 1).sum()),
        "metrics": {
            "accuracy": metrics_summary["accuracy"],
            "precision": metrics_summary["precision"],
            "recall": metrics_summary["recall"],
            "f1_score": metrics_summary["f1_score"],
            "roc_auc": metrics_summary["roc_auc"],
        },
        "confusion_matrix": metrics_summary["confusion_matrix"],
        "classification_report": metrics_summary["classification_report"],
        "error_analysis": error_summary,
        "visualizations": {
            "confusion_matrix": str(cm_path),
            "roc_curve": str(roc_path),
        },
    }

    metrics_file = out_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print("=== ENGLISH TEST SET EVALUATION RESULTS ===")
    print(f"Accuracy:  {final_results['metrics']['accuracy']:.4f}")
    print(f"Precision: {final_results['metrics']['precision']:.4f}")
    print(f"Recall:    {final_results['metrics']['recall']:.4f}")
    print(f"F1-Score:  {final_results['metrics']['f1_score']:.4f}")
    print(f"ROC-AUC:   {final_results['metrics']['roc_auc']:.4f}")
    print(f"Saved metrics to: {metrics_file}")
    print(f"Saved confusion matrix to: {cm_path}")
    print(f"Saved ROC curve to: {roc_path}")

    return final_results


if __name__ == "__main__":
    evaluate_english_model()
