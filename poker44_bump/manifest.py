"""Public model manifest for the poker-hot-0 Poker44 miner."""
from __future__ import annotations

import os
from typing import Any, Dict

MODEL_MANIFEST: Dict[str, Any] = {
    "schema_version": "1.0",
    "repo_url": "https://github.com/algodev999/poker-hot-0",
    "repo_commit": "fff7485",
    "model_name": "poker44-hot-0-bump",
    "model_version": "2026-07-02-hot-0",
    "framework": "scikit-learn/joblib",
    "license": "MIT",
    "training_data": "Private Poker44 benchmark-derived training data; no secrets or raw private data are published in this repo.",
    "data_handling": "Miner-visible payload features only; production evaluation data is supplied by Poker44 platform infrastructure.",
    "calibration_head": "conformal-maxhuman",
    "calibration_summary": "fixed-threshold conformal max-human calibration",
}


def get_model_manifest() -> Dict[str, Any]:
    """Return the manifest miners should attach to DetectionSynapse responses."""
    manifest = dict(MODEL_MANIFEST)
    repo_url = os.getenv("POKER44_MODEL_REPO_URL")
    repo_commit = os.getenv("POKER44_MODEL_REPO_COMMIT")
    if repo_url:
        manifest["repo_url"] = repo_url
    if repo_commit:
        manifest["repo_commit"] = repo_commit
    return manifest
