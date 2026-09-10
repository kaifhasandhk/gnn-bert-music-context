import numpy as np
from sklearn.metrics import f1_score

def evaluate_with_thresh(logits: np.ndarray, labels: np.ndarray, threshold: float):
    """Evaluates multi-label Macro-F1 and Micro-F1 scores at a target threshold."""
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    
    macro = f1_score(labels, preds, average='macro', zero_division=0)
    micro = f1_score(labels, preds, average='micro', zero_division=0)
    return macro, micro

def sweep_thresholds(probs: np.ndarray, labels: np.ndarray):
    """Sweeps thresholds [0.05..0.50] on validation set to find optimal Micro-F1."""
    best_thresh, best_micro = 0.15, -1.0
    for t in np.arange(0.05, 0.55, 0.05):
        preds = (probs >= t).astype(int)
        micro = f1_score(labels, preds, average='micro', zero_division=0)
        if micro > best_micro:
            best_micro = micro
            best_thresh = t
    return best_thresh, best_micro
