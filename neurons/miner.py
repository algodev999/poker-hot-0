"""Poker44 miner — hot-0 · CONFORMAL max-human calibration.

Tree ensemble (LightGBM + XGBoost + ExtraTrees + RandomForest) over signature-
collision chunk features, with a fixed-threshold conformal head (see
poker44_bump/head.py). The decision threshold T is fitted at train time to sit
just above the worst recent human score, so chunk FPR stays under the validator's
10% cliff. The model artifact is hot-reloaded whenever the daily retrain job
rewrites it — no restart needed.

Wallet / hotkey / axon port are supplied on the command line by the pm2
ecosystem config, which reads them from this project's .env.
"""
# NOTE: no `from __future__ import annotations` — bittensor's axon.attach()
# introspects forward()'s annotations at runtime with issubclass().
import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Tuple

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))  # prefer this project's local poker44 / poker44_bump

import bittensor as bt
import joblib

from poker44.base.miner import BaseMinerNeuron
from poker44.validator.synapse import DetectionSynapse
from poker44_bump import head as scoring_head


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


class ConformalMiner(BaseMinerNeuron):
    """Fixed-T conformal miner (hot-0)."""

    def __init__(self, config=None):
        super().__init__(config=config)
        self.model_path = Path(os.getenv("POKER44_BUMP_MODEL", str(_ROOT / "models" / "bump_model.joblib")))
        self._model = None
        self._model_mtime = 0.0
        self._load_model()
        bt.logging.info(
            f"🂡 hot-0 CONFORMAL miner up | head={scoring_head.HEAD_NAME} "
            f"T={self._model.threshold:.4f} feats={len(self._model.feature_names)} "
            f"oof_ap={self._model.metadata.get('oof_ap')}"
        )

    # ---- model lifecycle (hot-reload on daily retrain) ----
    def _load_model(self) -> None:
        self._model = joblib.load(self.model_path)
        self._model_mtime = self.model_path.stat().st_mtime
        md = self._model.metadata
        self.model_manifest = {
            "schema_version": "1",
            "open_source": True,
            "repo_url": os.getenv("POKER44_MODEL_REPO_URL", ""),
            "model_name": "poker44-conformal-maxhuman",
            "model_version": md.get("model_version", "conformal-v1"),
            "framework": "tree-ensemble+conformal-maxhuman",
            "license": "MIT",
            "inference_mode": "local-joblib",
            "scoring_head": scoring_head.HEAD_NAME,
            "training_data_statement": (
                f"Trained on {md.get('benchmark_rows', '?')} released benchmark chunks; "
                f"fixed-T conformal head T={md.get('conformal_threshold')}."
            ),
            "training_data_sources": md.get("training_data_sources") or ["released_training_benchmark"],
            "private_data_attestation": "No validator-private data used; released benchmark labels only.",
            "artifact_sha256": _sha256(self.model_path),
            "oof_ap": md.get("oof_ap"),
        }

    def _maybe_reload(self) -> None:
        try:
            mtime = self.model_path.stat().st_mtime
            if mtime > self._model_mtime:
                bt.logging.info("detected refreshed artifact — hot-reloading model")
                self._load_model()
        except FileNotFoundError:
            pass

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        self._maybe_reload()
        chunks = [list(c or []) for c in (synapse.chunks or [])]
        t0 = time.perf_counter()
        try:
            raw = self._model.predict_raw(chunks)
            scores = scoring_head.score_batch(raw, threshold=self._model.threshold)
        except Exception as err:
            bt.logging.warning(f"scoring failed ({err}); returning 0.5 for all chunks")
            scores = [0.5] * len(chunks)
        scores = [max(0.0, min(1.0, float(s))) for s in scores]
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        dt = (time.perf_counter() - t0) * 1000.0
        bt.logging.info(
            f"[hot-0/conformal] scored {len(chunks)} chunks in {dt:.1f}ms | "
            f"bots={sum(synapse.predictions)} "
            f"range=[{min(scores) if scores else 0:.3f},{max(scores) if scores else 0:.3f}]"
        )
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with ConformalMiner() as miner:
        bt.logging.info("hot-0 conformal miner running...")
        while True:
            bt.logging.info(f"UID {miner.uid} | Incentive {miner.metagraph.I[miner.uid]}")
            time.sleep(300)
