import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

#returns result of precision,recall and f1 values based on different thresholds
def threshold_search(y_true, probs, thresholds):
    rows = []

    for th in thresholds:
      #1 -> 확률이 >= th일때  pos_label: 불량, -1 -> neg_label: 정상
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