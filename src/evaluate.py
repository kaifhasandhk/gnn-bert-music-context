import torch
import numpy as np
from sklearn.metrics import f1_score

def evaluate_predictions(y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.3):
    """Calculates Micro-F1 and Macro-F1 metrics for multi-label tag prediction."""
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    
    print(f"Evaluation Results (Threshold={threshold}):")
    print(f" -> Macro-F1: {macro_f1:.3f}")
    print(f" -> Micro-F1: {micro_f1:.3f}")
    
    return {"macro_f1": macro_f1, "micro_f1": micro_f1}

if __name__ == "__main__":
    # Dummy verification run
    y_true_sample = np.random.randint(0, 2, size=(100, 50))
    y_pred_sample = np.random.rand(100, 50)
    evaluate_predictions(y_true_sample, y_pred_sample)
