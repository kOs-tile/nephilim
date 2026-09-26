import copy

import pytest

from nephilim.classifier.sandwich_detector import SandwichDetector
from nephilim.evidence import (
    build_sandwich_evidence,
    fingerprint_detector_input,
    verify_detector_evidence,
)
from nephilim.stream.transaction_decoder import TxRecord


TOKEN_A = "0x1111111111111111111111111111111111111111"
TOKEN_B = "0x2222222222222222222222222222222222222222"
BOT = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
VICTIM = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def tx(hash_, position, sender, gas_gwei, *, reverse=False, block=19_000_001):
    return TxRecord(
        hash=hash_,
        block_number=block,
        position_in_block=position,
        from_address=sender,
        to_address="0xe592427a0aece92de3edee1f18e0157c05861564",
        value_wei=int(5e18),
        value_eth=5.0,
        value_usd=0.0,
        gas_price_wei=int(gas_gwei * 1e9),
        gas_used=150_000,
        input_data="0x414bf389",
        method_selector="0x414bf389",
        is_swap=True,
        dex_pair_key=f"uniswap_v3:{TOKEN_A}:{TOKEN_B}:3000",
        dex_token_in=TOKEN_B if reverse else TOKEN_A,
        dex_token_out=TOKEN_A if reverse else TOKEN_B,
    )


def sandwich_block():
    return [
        tx("0xfront", 0, BOT, 45),
        tx("0xvictim", 1, VICTIM, 20),
        tx("0xback", 2, BOT, 45, reverse=True),
    ]


def test_detector_input_fingerprint_is_order_deterministic():
    txs = sandwich_block()

    assert fingerprint_detector_input(txs) == fingerprint_detector_input(list(reversed(txs)))


def test_sandwich_evidence_binds_detector_input_and_output():
    txs = sandwich_block()
    events = SandwichDetector().detect(txs)
    artifact = build_sandwich_evidence(txs, events)

    assert len(events) == 1
    assert artifact["event_count"] == 1
    assert len(artifact["input_fingerprint"]) == 64
    assert len(artifact["report_fingerprint"]) == 64
    assert artifact["validated_trading_signal"] is False
    assert artifact["authorizes_action"] is False
    assert artifact["economics"] == "heuristic_estimate"
    assert verify_detector_evidence(artifact)["valid"] is True


def test_event_order_does_not_change_report_fingerprint():
    txs = sandwich_block()
    event = SandwichDetector().detect(txs)[0]
    second = copy.deepcopy(event)
    second["tx_hashes"] = ["0xf2", "0xv2", "0xb2"]

    first_artifact = build_sandwich_evidence(txs, [event, second])
    second_artifact = build_sandwich_evidence(txs, [second, event])

    assert first_artifact["report_fingerprint"] == second_artifact["report_fingerprint"]


def test_detector_input_change_changes_report_fingerprint():
    txs = sandwich_block()
    events = SandwichDetector().detect(txs)
    first = build_sandwich_evidence(txs, events)

    changed = sandwich_block()
    changed[1].gas_price_wei = int(19e9)
    changed_events = SandwichDetector().detect(changed)
    second = build_sandwich_evidence(changed, changed_events)

    assert first["input_fingerprint"] != second["input_fingerprint"]
    assert first["report_fingerprint"] != second["report_fingerprint"]


def test_tampered_detector_evidence_fails_integrity():
    txs = sandwich_block()
    artifact = build_sandwich_evidence(txs, SandwichDetector().detect(txs))
    artifact["events"][0]["pair"] = "tampered"

    verification = verify_detector_evidence(artifact)

    assert verification["valid"] is False
    assert any(
        check["name"] == "integrity" and not check["ok"]
        for check in verification["checks"]
    )


def test_cross_block_input_fails_closed():
    txs = sandwich_block()
    txs[-1].block_number += 1

    with pytest.raises(ValueError, match="exactly one block"):
        build_sandwich_evidence(txs, [])
