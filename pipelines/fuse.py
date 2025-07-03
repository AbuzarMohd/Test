import numpy as np
LABELS = ["negative", "positive"]

def fuse(modal):
    weights = {k: v.max() for k, v in modal.items()}
    total   = sum(weights.values()) or 1
    fused   = sum(weights[k]*modal[k] for k in modal)/total
    return int(np.argmax(fused)), fused
