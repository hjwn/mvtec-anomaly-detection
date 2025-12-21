from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score

def auroc(y_true, y_score) -> float:
    """
    y_true: (N,) 0/1
    y_score: (N,) float (higher => more anomalous)
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    # roc_auc_score requires both classes present; handle edge cases safely
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))
