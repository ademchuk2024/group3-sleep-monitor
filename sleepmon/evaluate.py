"""
evaluate.py - Scores a model's stage predictions against the expert labels.

Everything is computed from one table, the confusion matrix: rows are the
expert's stage, columns are the model's stage, and each cell counts windows.

What each number means, for one stage (say Deep):
    precision  of the windows the model called Deep, what share really were Deep
    recall     of the windows that really were Deep, what share the model found
    F1         one number combining the two (their harmonic mean)
And for the whole model:
    accuracy   share of all windows it got right. Misleading on its own here:
               calling everything "Light" already scores over 50%.
    macro F1   the plain average of the four F1 scores, so the rare stages
               (Deep, REM) count as much as the common one. Our main score.
    kappa      Cohen's kappa: agreement with the expert beyond what chance
               alone would give. 0 = no better than chance, 1 = perfect.
"""

import numpy as np

import config

N = len(config.STAGES)


def confusion_matrix(truth, predicted, n=N):
    """Count windows by (expert stage, model stage). Returns an (n, n) table of whole numbers."""
    truth = np.asarray(truth, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    if truth.shape != predicted.shape:
        raise ValueError("truth and predicted must have the same length.")
    if len(truth) and (truth.min() < 0 or truth.max() >= n or predicted.min() < 0 or predicted.max() >= n):
        raise ValueError(f"stages must be whole numbers from 0 to {n - 1}.")
    return np.bincount(truth * n + predicted, minlength=n * n).reshape(n, n)


def scores(matrix):
    """All the scores from a confusion matrix, as a plain dict (ready to save as JSON)."""
    matrix = np.asarray(matrix, dtype=np.float64)
    total = matrix.sum()
    correct = np.diag(matrix)
    truth_count = matrix.sum(axis=1)       # how many windows the expert gave each stage
    model_count = matrix.sum(axis=0)       # how many the model gave each stage

    def safe_divide(a, b):
        # 0 where there is nothing to divide by (for example a stage the model never predicts).
        return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b > 0)

    precision = safe_divide(correct, model_count)
    recall = safe_divide(correct, truth_count)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    accuracy = correct.sum() / total if total else 0.0
    # Agreement expected by chance: if model and expert each picked stages at
    # their own usual rates, independently of each other.
    chance = float((truth_count * model_count).sum() / total ** 2) if total else 0.0
    kappa = (accuracy - chance) / (1 - chance) if chance < 1 else 0.0

    present = truth_count > 0              # average only over stages that occur in the data
    return {
        "windows": int(total),
        "accuracy": float(accuracy),
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "kappa": float(kappa),
        "per_stage": {
            stage: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
                    "windows": int(truth_count[i])}
            for i, stage in enumerate(config.STAGES)
        },
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def evaluate(model, nights):
    """Run a model on prepared nights and score it on the windows that have an expert stage.

    Returns {"overall": scores, "by_dataset": {name: scores}}.
    """
    overall = np.zeros((N, N), dtype=np.int64)
    by_dataset = {}
    for night in nights:
        if not night.scored.any():
            continue
        predicted = np.argmax(model.predict_proba(night)[night.scored], axis=1)
        matrix = confusion_matrix(night.y[night.scored], predicted)
        overall += matrix
        by_dataset[night.dataset] = by_dataset.get(night.dataset, 0) + matrix
    return {"overall": scores(overall), "by_dataset": {name: scores(m) for name, m in sorted(by_dataset.items())}}


def format_table(result, title=""):
    """A small text table of one evaluation, for printing."""
    lines = [title] if title else []
    lines.append(f"{'':10s} {'windows':>8s} {'accuracy':>8s} {'macro F1':>8s} {'kappa':>6s} | "
                 + " ".join(f"{s + ' rec':>10s}" for s in config.STAGES))
    for name, s in [("ALL", result["overall"])] + list(result["by_dataset"].items()):
        recalls = " ".join(f"{s['per_stage'][stage]['recall']:10.2f}" for stage in config.STAGES)
        lines.append(f"{name:10s} {s['windows']:8d} {s['accuracy']:8.3f} {s['macro_f1']:8.3f} {s['kappa']:6.3f} | {recalls}")
    return "\n".join(lines)
