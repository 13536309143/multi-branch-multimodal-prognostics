import numpy as np


def regression_metrics(target, prediction):
    target, prediction = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if target.ndim != 1 or target.shape != prediction.shape or not len(target):
        raise ValueError("Metrics require nonempty matching vectors")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all() or np.any(target <= 0):
        raise ValueError("MRE requires finite predictions and strictly positive targets")
    error = prediction - target
    return {"mae": float(np.mean(np.abs(error))), "mse": float(np.mean(error**2)),
            "mre_percent": float(100 * np.mean(np.abs(error) / target)),
            "rmse": float(np.sqrt(np.mean(error**2))), "n": len(target)}


def ensemble_weights(errors, mode="inverse_mae"):
    errors = np.asarray(errors, dtype=float)
    if errors.shape != (3,) or not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("Expected three nonnegative validation errors")
    if mode == "uniform":
        return np.ones(3) / 3
    if mode != "inverse_mae":
        raise ValueError("Unknown ensemble weighting")
    if np.any(errors == 0):
        weights = (errors == 0).astype(float)
    else:
        weights = 1.0 / errors
    return weights / weights.sum()
