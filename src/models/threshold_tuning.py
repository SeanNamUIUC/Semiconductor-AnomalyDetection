import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def threshold_search(y_true, probs, thresholds):
    rows = []

    for th in thresholds:
        pred = np.where(probs >= th, 1, -1)

        rows.append({
            "threshold": th,
            "precision": precision_score(
                y_true, pred,
                pos_label=1,
                zero_division=0
            ),
            "recall": recall_score(
                y_true, pred,
                pos_label=1,
                zero_division=0
            ),
            "f1": f1_score(
                y_true, pred,
                pos_label=1,
                zero_division=0
            )
        })

    return pd.DataFrame(rows)