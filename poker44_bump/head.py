"""hot-0 scoring head — CONFORMAL max-human (fixed-T monotone map).

This miner's calibration philosophy: pin an absolute decision threshold `T` just
above the worst recent *human* raw score (fit at train time, stored on the
artifact). A hand-batch crosses 0.5 only when its raw bot-probability exceeds
that human ceiling, so chunk-level FPR stays under the validator's 10% cliff
regardless of how many bots happen to be in a given query window.

Because the map is a fixed monotone function of the raw score (not batch-
relative), two identical chunks always get identical scores — this head is fully
deterministic and query-order independent. That is the deliberate contrast with
the sibling miners (hot-1 batch-relative top-k, hot-2 percentile logistic).
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

HEAD_NAME = "conformal-maxhuman"


def score_batch(raw: Sequence[float], *, threshold: float,
                lo: float = 0.02, hi: float = 0.98) -> List[float]:
    """Map raw bot-probabilities through the fixed-T conformal curve.

    p == T -> 0.5 ; p < T -> [lo, 0.5) ; p > T -> (0.5, hi].
    Ranking (AP) is invariant to this monotone map, so calibration only moves
    the FPR/recall operating point, never the ranker's quality.
    """
    p = np.clip(np.asarray(raw, dtype=np.float64), 0.0, 1.0)
    T = float(min(max(threshold, 1e-4), 1.0 - 1e-4))
    below = 0.5 * (p / T)
    above = 0.5 + 0.5 * (p - T) / (1.0 - T)
    out = np.where(p >= T, above, below)
    return [float(v) for v in np.clip(out, lo, hi)]
