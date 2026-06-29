"""
nephilim.classifier.mev_classifier
------------------------------------
XGBoost-based classifier for 12 MEV transaction types.

Feature vector
--------------
The model operates on 6 numeric features extracted from each ``TxRecord``:

1. ``gas_price_percentile``     — 0–100, relative to the block
2. ``position_in_block``        — absolute index in block
3. ``involves_flashloan``       — binary 0/1
4. ``touches_price_oracle``     — binary 0/1
5. ``same_block_counterpart``   — binary 0/1
6. ``value_extracted_eth``      — estimated ETH value involved

Plus rule-based overrides for high-confidence pattern-detected cases (sandwich,
JIT) that have already been flagged by the dedicated detectors.

MEV types
---------
sandwich_attack, jit_liquidity, cex_dex_arb, pure_arb, liquidation,
nft_sweep, wash_trade, bridge, organic_swap, flashloan_attack,
oracle_manipulation, governance_attack
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from loguru import logger

from nephilim.stream.transaction_decoder import TxRecord

# Canonical label ordering (must match training)
MEV_TYPES: List[str] = [
    "sandwich_attack",
    "jit_liquidity",
    "cex_dex_arb",
    "pure_arb",
    "liquidation",
    "nft_sweep",
    "wash_trade",
    "bridge",
    "organic_swap",
    "flashloan_attack",
    "oracle_manipulation",
    "governance_attack",
]

FEATURE_NAMES: List[str] = [
    "gas_price_percentile",
    "position_in_block",
    "involves_flashloan",
    "touches_price_oracle",
    "same_block_counterpart",
    "value_extracted_eth",
    "gas_premium_multiplier",
    "is_swap",
    "is_lp_add",
    "is_lp_remove",
    "is_governance",
]


def extract_features(tx: TxRecord) -> np.ndarray:
    """Build the feature vector from a TxRecord."""
    return np.array(
        [
            tx.gas_price_percentile,
            tx.position_in_block,
            float(tx.involves_flashloan),
            float(tx.touches_price_oracle),
            float(tx.same_block_counterpart),
            tx.value_eth,  # proxy for extracted value
            tx.gas_premium_multiplier,
            float(tx.is_swap),
            float(tx.is_lp_add),
            float(tx.is_lp_remove),
            float(tx.is_governance),
        ],
        dtype=np.float32,
    )


class MEVClassifier:
    """
    Loads a trained XGBoost model and classifies each TxRecord.

    Falls back to a rule-based classifier when the model file is absent
    (e.g. before the first training run).
    """

    def __init__(
        self,
        model_path: Path = Path("models/mev_classifier.pkl"),
        label_encoder_path: Optional[Path] = None,
    ) -> None:
        self._model = None
        self._label_encoder = None
        self._model_path = model_path
        self._label_encoder_path = label_encoder_path or Path(
            str(model_path).replace("mev_classifier", "label_encoder")
        )
        self._load_model()

    def _load_model(self) -> None:
        if self._model_path.exists():
            try:
                self._model = joblib.load(self._model_path)
                self._label_encoder = joblib.load(self._label_encoder_path)
                logger.info("MEVClassifier loaded model from {}", self._model_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load model: {} — using rule-based fallback", exc)
        else:
            logger.warning(
                "Model file {} not found — using rule-based fallback. "
                "Run `python nephilim/ml/train_classifier.py` to train.",
                self._model_path,
            )

    def predict(self, tx: TxRecord) -> TxRecord:
        """
        Classify the transaction in-place (sets ``mev_type`` and
        ``mev_confidence``) and return the same object.
        """
        # Rule-based override for already-detected patterns
        if "sandwich" in tx.pattern_flags:
            tx.mev_type = "sandwich_attack"
            tx.mev_confidence = 0.95
            return tx
        if "jit_liquidity" in tx.pattern_flags:
            tx.mev_type = "jit_liquidity"
            tx.mev_confidence = 0.95
            return tx

        if self._model is not None:
            mev_type, confidence = self._ml_predict(tx)
        else:
            mev_type, confidence = self._rule_predict(tx)

        tx.mev_type = mev_type
        tx.mev_confidence = confidence
        return tx

    def _ml_predict(self, tx: TxRecord) -> tuple[str, float]:
        features = extract_features(tx).reshape(1, -1)
        proba = self._model.predict_proba(features)[0]
        class_idx = int(np.argmax(proba))
        confidence = float(proba[class_idx])
        label = self._label_encoder.inverse_transform([class_idx])[0]
        return label, confidence

    def _rule_predict(self, tx: TxRecord) -> tuple[str, float]:
        """Deterministic rule-based fallback."""
        if tx.involves_flashloan and tx.touches_price_oracle:
            return "oracle_manipulation", 0.80
        if tx.involves_flashloan:
            return "flashloan_attack", 0.75
        if tx.is_governance:
            return "governance_attack", 0.82
        if tx.touches_price_oracle:
            return "oracle_manipulation", 0.65
        if tx.is_lp_add and tx.is_lp_remove:
            return "jit_liquidity", 0.85
        if tx.gas_price_percentile >= 95 and tx.is_swap:
            return "sandwich_attack", 0.70
        if tx.gas_price_percentile >= 85 and tx.is_swap:
            return "cex_dex_arb", 0.60
        if tx.is_swap and tx.value_eth >= 5.0:
            return "pure_arb", 0.55
        if tx.is_swap:
            return "organic_swap", 0.80
        return "organic_swap", 0.60

    def predict_batch(self, txs: List[TxRecord]) -> List[TxRecord]:
        """Vectorised batch prediction (more efficient for replay scenarios)."""
        if not txs:
            return txs

        # Apply rule overrides first
        remaining = []
        for tx in txs:
            if "sandwich" in tx.pattern_flags:
                tx.mev_type = "sandwich_attack"
                tx.mev_confidence = 0.95
            elif "jit_liquidity" in tx.pattern_flags:
                tx.mev_type = "jit_liquidity"
                tx.mev_confidence = 0.95
            else:
                remaining.append(tx)

        if not remaining or self._model is None:
            for tx in remaining:
                mev_type, conf = self._rule_predict(tx)
                tx.mev_type = mev_type
                tx.mev_confidence = conf
            return txs

        # Batch ML prediction
        X = np.array([extract_features(tx) for tx in remaining], dtype=np.float32)
        probas = self._model.predict_proba(X)
        for tx, proba in zip(remaining, probas):
            class_idx = int(np.argmax(proba))
            tx.mev_confidence = float(proba[class_idx])
            tx.mev_type = self._label_encoder.inverse_transform([class_idx])[0]

        return txs
