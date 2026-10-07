import numpy as np


def pd_to_score(pd_array, target_score: int = 600, target_odds: float = 50, pdo: float = 20) -> np.ndarray:
    """Higher score = lower risk. `target_odds` (good:bad) scores `target_score`; odds double every `pdo` points."""
    pd_array = np.clip(np.asarray(pd_array, dtype=float), 1e-6, 1 - 1e-6)
    factor = pdo / np.log(2)
    offset = target_score - factor * np.log(target_odds)
    return np.round(offset - factor * np.log(pd_array / (1 - pd_array))).astype(int)