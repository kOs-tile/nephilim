"""Deterministic detector-evidence artifacts for NEPHILIM.

Detector output is research evidence, not execution or trading authority.
The artifact binds a detector result to the exact normalized transaction inputs
used by the detector so downstream KAVI audit envelopes can reference it by digest.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from nephilim.stream.transaction_decoder import TxRecord


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _normalized_tx(tx: TxRecord) -> dict[str, Any]:
    """Keep only detector-relevant, reproducible transaction evidence."""
    return {
        "hash": tx.hash,
        "block_number": tx.block_number,
        "position_in_block": tx.position_in_block,
        "from_address": tx.from_address,
        "to_address": tx.to_address,
        "method_selector": tx.method_selector,
        "gas_price_wei": tx.gas_price_wei,
        "gas_used": tx.gas_used,
        "value_wei": tx.value_wei,
        "dex_pair_key": tx.dex_pair_key,
        "dex_token_in": tx.dex_token_in,
        "dex_token_out": tx.dex_token_out,
    }


def fingerprint_detector_input(txs: Iterable[TxRecord]) -> str:
    normalized = [_normalized_tx(tx) for tx in txs]
    normalized.sort(key=lambda row: (row["block_number"], row["position_in_block"], row["hash"]))
    return _digest(normalized)


def build_sandwich_evidence(
    txs: Iterable[TxRecord],
    events: list[dict[str, Any]],
    *,
    chain_id: str = "ethereum",
) -> dict[str, Any]:
    """Seal sandwich-detector output as non-authoritative audit evidence."""
    tx_list = list(txs)
    if not tx_list:
        raise ValueError("txs must contain at least one transaction")

    blocks = {tx.block_number for tx in tx_list}
    if len(blocks) != 1:
        raise ValueError("sandwich evidence must bind to exactly one block")
    block_number = next(iter(blocks))

    canonical_events = sorted(
        [dict(event) for event in events],
        key=lambda event: _canonical(event),
    )

    artifact: dict[str, Any] = {
        "version": "nephilim.detector-evidence.v0",
        "artifact_id": f"{chain_id}:{block_number}:sandwich",
        "chain_id": str(chain_id),
        "block_number": block_number,
        "detector": "sandwich",
        "detector_contract": "canonical-pair+direction+gas-ordering.v0",
        "input_fingerprint": fingerprint_detector_input(tx_list),
        "event_count": len(canonical_events),
        "events": canonical_events,
        "signal_status": "research_evidence",
        "validated_trading_signal": False,
        "authorizes_action": False,
        "economics": (
            "heuristic_estimate"
            if any(event.get("extracted_value_is_estimate") for event in canonical_events)
            else "not_reported"
        ),
    }
    artifact["report_fingerprint"] = _digest(artifact)
    return artifact


def verify_detector_evidence(artifact: dict[str, Any]) -> dict[str, Any]:
    """Verify artifact integrity only; historical-label validity is a separate benchmark."""
    body = dict(artifact)
    claimed = body.pop("report_fingerprint", None)
    checks = [
        ("version", body.get("version") == "nephilim.detector-evidence.v0"),
        ("integrity", claimed == _digest(body)),
        ("non_authority", body.get("authorizes_action") is False),
        ("not_validated_trading_signal", body.get("validated_trading_signal") is False),
    ]
    return {
        "valid": all(ok for _, ok in checks),
        "checks": [{"name": name, "ok": ok} for name, ok in checks],
    }
