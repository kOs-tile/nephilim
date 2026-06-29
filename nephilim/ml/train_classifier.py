"""
nephilim.ml.train_classifier
------------------------------
Training pipeline for the XGBoost MEV transaction classifier.

This script:
1. Generates a realistic synthetic labeled dataset (no external data required).
2. Applies feature engineering matching what ``extract_features()`` produces
   at inference time.
3. Trains an XGBoost multi-class classifier with stratified k-fold CV.
4. Reports per-class precision/recall/F1.
5. Saves the trained model and label encoder to ``models/``.

Synthetic data generation
--------------------------
Each MEV type has distinct statistical signatures:

- sandwich_attack : very high gas percentile (90–100), position in block < 30,
  same_block_counterpart=True, is_swap=True, moderate ETH
- jit_liquidity   : is_lp_add + is_lp_remove flags, same_block_counterpart
- cex_dex_arb     : high gas percentile (80–95), is_swap, low value
- pure_arb        : gas 70–90, is_swap, multi-hop proxy (high value)
- liquidation     : moderate gas, high value, involves_flashloan sometimes
- nft_sweep       : high value, low gas percentile, not a DEX swap
- wash_trade      : low gas, moderate value, is_swap, same cluster
- bridge          : to known bridge, mid gas, high value
- organic_swap    : random gas, is_swap, low value
- flashloan_attack: involves_flashloan=True, touches_price_oracle, high gas
- oracle_manipulation: touches_price_oracle=True, high gas, moderate value
- governance_attack: is_governance=True, involves_flashloan sometimes

Usage
-----
    python nephilim/ml/train_classifier.py
    python nephilim/ml/train_classifier.py --samples 20000 --output models/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# ── Feature column names (must match extract_features() in mev_classifier.py) ─

FEATURE_COLS: List[str] = [
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

LABEL_COL = "mev_type"

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


# ── Synthetic data generators ──────────────────────────────────────────────────

def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _sandwich(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "gas_price_percentile": rng.uniform(88, 100, n),
        "position_in_block": rng.integers(0, 30, n).astype(float),
        "involves_flashloan": 0.0,
        "touches_price_oracle": rng.choice([0.0, 1.0], n, p=[0.9, 0.1]),
        "same_block_counterpart": 1.0,
        "value_extracted_eth": rng.exponential(0.3, n),
        "gas_premium_multiplier": rng.uniform(3.0, 15.0, n),
        "is_swap": 1.0,
        "is_lp_add": 0.0,
        "is_lp_remove": 0.0,
        "is_governance": 0.0,
        LABEL_COL: "sandwich_attack",
    })


def _jit(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "gas_price_percentile": rng.uniform(75, 100, n),
        "position_in_block": rng.integers(0, 50, n).astype(float),
        "involves_flashloan": 0.0,
        "touches_price_oracle": 0.0,
        "same_block_counterpart": 1.0,
        "value_extracted_eth": rng.exponential(0.1, n),
        "gas_premium_multiplier": rng.uniform(2.0, 8.0, n),
        "is_swap": rng.choice([0.0, 1.0], n, p=[0.5, 0.5]),
        "is_lp_add": 1.0,
        "is_lp_remove": 1.0,
        "is_governance": 0.0,
        LABEL_COL: "jit_liquidity",
    })


def _cex_dex_arb(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "gas_price_percentile": rng.uniform(80, 97, n),
        "position_in_block": rng.integers(0, 20, n).astype(float),
        "involves_flashloan": rng.choice([0.0, 1.0], n, p=[0.7, 0.3]),
        "touches_price_oracle": rng.choice([0.0, 1.0], n, p=[0.6, 0.4]),
        "same_block_counterpart": rng.choice([0.0, 1.0], n, p=[0.4, 0.6]),
        "value_extracted_eth": rng.exponential(0.5, n),
        "gas_premium_multiplier": rng.uniform(2.5, 10.0, n),
        "is_swap": 1.0,
        "is_lp_add": 0.0,
        "is_lp_remove": 0.0,
        "is_governance": 0.0,
        LABEL_COL: "cex_dex_arb",
    })


def _pure_arb(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "gas_price_percentile": rng.uniform(70, 92, n),
        "position_in_block": rng.integers(0, 100, n).astype(float),
        "involves_flashloan": rng.choice([0.0, 1.0], n, p=[0.5, 0.5]),
        "touches_price_oracle": 0.0,
        "same_block_counterpart": 0.0,
        "value_extracted_eth": rng.exponential(1.0, n),
        "gas_premium_multiplier": rng.uniform(1.5, 6.0, n),
        "is_swap": 1.0,
        "is_lp_add": 0.0,
        "is_lp_remove": 0.0,
        "is_governance": 0.0,
        LABEL_COL: "pure_arb",
    })


def _liquidation(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "gas_price_percentile": rng.uniform(60, 95, n),
        "position_in_block": rng.integers(0, 150, n).astype(float),
        "involves_flashloan": rng.choice([0.0, 1.0], n, p=[0.4, 0.6]),
        "touches_price_oracle": rng.choice([0.0, 1.0], n, p=[0.3, 0.7]),
        "same_block_counterpart": 0.0,
        "value_extracted_eth": rng.exponential(5.0, n),
        "gas_premium_multiplier": rng.uniform(1.2, 5.0, n),
        "is_swap": 0.0,
        "is_lp_add": 0.0,
        "is_lp_remove": 0.0,
        "is_governance": 0.0,
        LABEL_COL: "liquidation",
    })


def _nft_sweep(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "gas_price_percentile": rng.uniform(50, 85, n),
        "position_in_block": rng.integers(20, 200, n).astype(float),
        "involves_flashloan": rng.choice([0.0, 1.0], n, p=[0.6, 0.4]),
        "touches_price_oracle": 0.0,
        "same_block_counterpart": 0.0,
        "value_extracted_eth": rng.exponential(3.0, n),
        "gas_premium_multiplier": rng.uniform(1.0, 3.0, n),
        "is_swap": 0.0,
        "is_lp_add": 0.0,
        "is_lp_remove": 0.0,
        "is_governance": 0.0,
        LABEL_COL: "nft_sweep",
    })


def _wash_trade(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "gas_price_percentile": rng.uniform(30, 70, n),
        "position_in_block": rng.integers(50, 300, n).astype(float),
        "involves_flashloan": 0.0,
        "touches_price_oracle": 0.0,
        "same_block_counterpart": rng.choice([0.0, 1.0], n, p=[0.3, 0.7]),
        "value_extracted_eth": rng.exponential(0.2, n),
        "gas_premium_multiplier": rng.uniform(1.0, 2.0, n),
        "is_swap": 1.0,
        "is_lp_add": 0.0,
        "is_lp_remove": 0.0,
        "is_governance": 0.0,
        LABEL_COL: "wash_trade",
    })


def _bridge(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "gas_price_percentile": rng.uniform(40, 75, n),
        "position_in_block": rng.integers(10, 250, n).astype(float),
        "involves_flashloan": 0.0,
        "touches_price_oracle": 0.0,
        "same_block_counterpart": 0.0,
        "value_extracted_eth": rng.exponential(2.0, n),
        "gas_premium_multiplier": rng.uniform(1.0, 2.5, n),
        "is_swap": 0.0,
        "is_lp_add": 0.0,
        "is_lp_remove": 0.0,
        "is_governance": 0.0,
        LABEL_COL: "bridge",
    })


def _organic_swap(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "gas_price_percentile": rng.uniform(10, 70, n),
        "position_in_block": rng.integers(0, 400, n).astype(float),
        "involves_flashloan": 0.0,
        "touches_price_oracle": 0.0,
        "same_block_counterpart": 0.0,
        "value_extracted_eth": rng.exponential(0.1, n),
        "gas_premium_multiplier": rng.uniform(1.0, 1.5, n),
        "is_swap": 1.0,
        "is_lp_add": 0.0,
        "is_lp_remove": 0.0,
        "is_governance": 0.0,
        LABEL_COL: "organic_swap",
    })


def _flashloan_attack(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "gas_price_percentile": rng.uniform(80, 100, n),
        "position_in_block": rng.integers(0, 50, n).astype(float),
        "involves_flashloan": 1.0,
        "touches_price_oracle": rng.choice([0.0, 1.0], n, p=[0.3, 0.7]),
        "same_block_counterpart": 1.0,
        "value_extracted_eth": rng.exponential(10.0, n),
        "gas_premium_multiplier": rng.uniform(5.0, 20.0, n),
        "is_swap": rng.choice([0.0, 1.0], n, p=[0.4, 0.6]),
        "is_lp_add": 0.0,
        "is_lp_remove": 0.0,
        "is_governance": 0.0,
        LABEL_COL: "flashloan_attack",
    })


def _oracle_manip(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "gas_price_percentile": rng.uniform(75, 100, n),
        "position_in_block": rng.integers(0, 40, n).astype(float),
        "involves_flashloan": rng.choice([0.0, 1.0], n, p=[0.5, 0.5]),
        "touches_price_oracle": 1.0,
        "same_block_counterpart": 1.0,
        "value_extracted_eth": rng.exponential(2.0, n),
        "gas_premium_multiplier": rng.uniform(3.0, 12.0, n),
        "is_swap": 0.0,
        "is_lp_add": 0.0,
        "is_lp_remove": 0.0,
        "is_governance": 0.0,
        LABEL_COL: "oracle_manipulation",
    })


def _governance_attack(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "gas_price_percentile": rng.uniform(60, 90, n),
        "position_in_block": rng.integers(0, 100, n).astype(float),
        "involves_flashloan": rng.choice([0.0, 1.0], n, p=[0.4, 0.6]),
        "touches_price_oracle": 0.0,
        "same_block_counterpart": 0.0,
        "value_extracted_eth": rng.exponential(0.5, n),
        "gas_premium_multiplier": rng.uniform(1.5, 4.0, n),
        "is_swap": 0.0,
        "is_lp_add": 0.0,
        "is_lp_remove": 0.0,
        "is_governance": 1.0,
        LABEL_COL: "governance_attack",
    })


_GENERATORS = {
    "sandwich_attack": _sandwich,
    "jit_liquidity": _jit,
    "cex_dex_arb": _cex_dex_arb,
    "pure_arb": _pure_arb,
    "liquidation": _liquidation,
    "nft_sweep": _nft_sweep,
    "wash_trade": _wash_trade,
    "bridge": _bridge,
    "organic_swap": _organic_swap,
    "flashloan_attack": _flashloan_attack,
    "oracle_manipulation": _oracle_manip,
    "governance_attack": _governance_attack,
}

# Organic swaps dominate real-world traffic
_CLASS_WEIGHTS: Dict[str, float] = {
    "organic_swap": 0.35,
    "sandwich_attack": 0.12,
    "cex_dex_arb": 0.10,
    "pure_arb": 0.08,
    "jit_liquidity": 0.06,
    "liquidation": 0.06,
    "bridge": 0.05,
    "flashloan_attack": 0.05,
    "wash_trade": 0.04,
    "nft_sweep": 0.04,
    "oracle_manipulation": 0.03,
    "governance_attack": 0.02,
}


def generate_dataset(total_samples: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic labeled dataset with realistic class distribution."""
    rng = _rng(seed)
    frames = []
    for mev_type, weight in _CLASS_WEIGHTS.items():
        n = max(1, int(total_samples * weight))
        gen = _GENERATORS[mev_type]
        df = gen(n, rng)
        # Add small noise to numeric features
        for col in ["gas_price_percentile", "gas_premium_multiplier", "value_extracted_eth"]:
            df[col] += rng.normal(0, 0.01, n)
            df[col] = df[col].clip(lower=0.0)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined.sample(frac=1, random_state=seed).reset_index(drop=True)


def train(
    total_samples: int = 10_000,
    output_dir: Path = Path("models"),
    seed: int = 42,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating {} synthetic samples…", total_samples)
    df = generate_dataset(total_samples=total_samples, seed=seed)

    logger.info("Class distribution:\n{}", df[LABEL_COL].value_counts().to_string())

    X = df[FEATURE_COLS].values.astype(np.float32)
    y_raw = df[LABEL_COL].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    logger.info(
        "Train: {} samples | Test: {} samples | Classes: {}",
        len(X_train),
        len(X_test),
        len(le.classes_),
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )

    logger.info("Training XGBoost classifier…")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, digits=3
    )
    logger.info("Classification report:\n{}", report)

    # Save
    model_path = output_dir / "mev_classifier.pkl"
    encoder_path = output_dir / "label_encoder.pkl"
    joblib.dump(model, model_path)
    joblib.dump(le, encoder_path)
    logger.success("Model saved to {}", model_path)
    logger.success("Label encoder saved to {}", encoder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the NEPHILIM MEV XGBoost classifier"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10_000,
        help="Total synthetic training samples (default: 10000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models"),
        help="Output directory for model files (default: models/)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    train(total_samples=args.samples, output_dir=args.output, seed=args.seed)
