"""
tests/test_sandwich_detector.py
----------------------------------
Unit tests for the SandwichDetector.

Tests verify:
1. A clean (frontrun, victim, backrun) triple is detected.
2. Triples with wrong gas ordering are not flagged.
3. Triples from different addresses are not flagged as a sandwich.
4. An attacker cannot sandwich themselves (victim must differ).
5. Extracted value is non-negative.
6. Multiple sandwiches in one block are all found.
"""

from __future__ import annotations

from typing import List

import pytest

from nephilim.classifier.sandwich_detector import SandwichDetector
from nephilim.stream.transaction_decoder import TxRecord

UNISWAP_V3 = "0xe592427a0aece92de3edee1f18e0157c05861564"
BOT_A = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
BOT_B = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
VICTIM = "0x1111111111111111111111111111111111111111"
BASE_FEE = int(20e9)


def make_tx(
    hash: str,
    position: int,
    from_addr: str,
    gas_gwei: float,
    value_eth: float = 5.0,
    to_addr: str = UNISWAP_V3,
    block: int = 19_000_000,
    all_gas_prices: List[int] | None = None,
) -> TxRecord:
    gas_wei = int(gas_gwei * 1e9)
    prices = all_gas_prices or [gas_wei]
    below = sum(1 for g in prices if g <= gas_wei)
    percentile = 100.0 * below / len(prices)
    return TxRecord(
        hash=hash,
        block_number=block,
        position_in_block=position,
        from_address=from_addr,
        to_address=to_addr,
        value_wei=int(value_eth * 1e18),
        value_eth=value_eth,
        value_usd=value_eth * 3500.0,
        gas_price_wei=gas_wei,
        gas_used=150_000,
        input_data="0x414bf389" + "00" * 32,
        method_selector="0x414bf389",
        is_swap=True,
        gas_price_percentile=percentile,
        block_base_fee_wei=BASE_FEE,
        gas_premium_multiplier=gas_wei / BASE_FEE,
    )


class TestSandwichDetector:
    def setup_method(self) -> None:
        self.detector = SandwichDetector()

    def _block_prices(self, *gas_gwei: float) -> List[int]:
        return [int(g * 1e9) for g in gas_gwei]

    def test_detects_canonical_sandwich(self) -> None:
        """A clear frontrun → victim → backrun triple is detected."""
        prices = self._block_prices(40, 40, 20, 19, 18)
        txs = [
            make_tx("0xfront", 0, BOT_A, 40.0, all_gas_prices=prices),
            make_tx("0xvictim", 1, VICTIM, 20.0, all_gas_prices=prices),
            make_tx("0xback", 2, BOT_A, 40.0, all_gas_prices=prices),
        ]
        results = self.detector.detect(txs)
        assert len(results) == 1
        attack = results[0]
        assert attack["attacker"] == BOT_A
        assert attack["victim"] == VICTIM
        assert "0xfront" in attack["tx_hashes"]
        assert "0xvictim" in attack["tx_hashes"]
        assert "0xback" in attack["tx_hashes"]

    def test_extracted_value_non_negative(self) -> None:
        """Extracted value should never be negative."""
        prices = self._block_prices(50, 50, 20)
        txs = [
            make_tx("0xfront", 0, BOT_A, 50.0, value_eth=0.01, all_gas_prices=prices),
            make_tx("0xvictim", 1, VICTIM, 20.0, value_eth=0.01, all_gas_prices=prices),
            make_tx("0xback", 2, BOT_A, 50.0, value_eth=0.01, all_gas_prices=prices),
        ]
        results = self.detector.detect(txs)
        if results:
            assert results[0]["extracted_value_eth"] >= 0.0

    def test_no_detection_same_gas(self) -> None:
        """
        If attacker and victim have the same gas price (ratio < MIN_GAS_RATIO),
        no sandwich is detected.
        """
        prices = self._block_prices(20, 20, 20)
        txs = [
            make_tx("0xfront", 0, BOT_A, 20.0, all_gas_prices=prices),
            make_tx("0xvictim", 1, VICTIM, 20.0, all_gas_prices=prices),
            make_tx("0xback", 2, BOT_A, 20.0, all_gas_prices=prices),
        ]
        results = self.detector.detect(txs)
        assert len(results) == 0

    def test_no_detection_attacker_victim_same_address(self) -> None:
        """An attacker cannot be their own victim."""
        prices = self._block_prices(45, 20, 45)
        txs = [
            make_tx("0xfront", 0, BOT_A, 45.0, all_gas_prices=prices),
            make_tx("0xvictim", 1, BOT_A, 20.0, all_gas_prices=prices),  # same
            make_tx("0xback", 2, BOT_A, 45.0, all_gas_prices=prices),
        ]
        results = self.detector.detect(txs)
        assert len(results) == 0

    def test_no_detection_wrong_backrun_address(self) -> None:
        """Backrun from a different address than frontrun is not a sandwich."""
        prices = self._block_prices(45, 20, 45)
        txs = [
            make_tx("0xfront", 0, BOT_A, 45.0, all_gas_prices=prices),
            make_tx("0xvictim", 1, VICTIM, 20.0, all_gas_prices=prices),
            make_tx("0xback", 2, BOT_B, 45.0, all_gas_prices=prices),  # different bot
        ]
        results = self.detector.detect(txs)
        assert len(results) == 0

    def test_multiple_sandwiches_same_block(self) -> None:
        """Two independent sandwiches on different pools are both detected."""
        pool_a = "0xaaaa000000000000000000000000000000000000"
        pool_b = "0xbbbb000000000000000000000000000000000000"
        prices = self._block_prices(50, 50, 20, 20, 50, 50)

        txs = [
            # Sandwich 1 on pool_a by BOT_A
            make_tx("0xf1", 0, BOT_A, 50.0, to_addr=pool_a, all_gas_prices=prices),
            make_tx("0xv1", 1, VICTIM, 20.0, to_addr=pool_a, all_gas_prices=prices),
            make_tx("0xb1", 2, BOT_A, 50.0, to_addr=pool_a, all_gas_prices=prices),
            # Sandwich 2 on pool_b by BOT_B
            make_tx("0xf2", 3, BOT_B, 50.0, to_addr=pool_b, all_gas_prices=prices),
            make_tx("0xv2", 4, VICTIM, 20.0, to_addr=pool_b, all_gas_prices=prices),
            make_tx("0xb2", 5, BOT_B, 50.0, to_addr=pool_b, all_gas_prices=prices),
        ]
        results = self.detector.detect(txs)
        assert len(results) == 2
        attackers = {r["attacker"] for r in results}
        assert BOT_A in attackers
        assert BOT_B in attackers

    def test_pattern_flags_set(self) -> None:
        """The frontrun, victim and backrun txs should have 'sandwich' pattern flag."""
        prices = self._block_prices(45, 20, 45)
        front = make_tx("0xfront", 0, BOT_A, 45.0, all_gas_prices=prices)
        victim = make_tx("0xvictim", 1, VICTIM, 20.0, all_gas_prices=prices)
        back = make_tx("0xback", 2, BOT_A, 45.0, all_gas_prices=prices)

        results = self.detector.detect([front, victim, back])
        assert len(results) == 1
        # same_block_counterpart should be marked
        assert front.same_block_counterpart is True
        assert victim.same_block_counterpart is True
        assert back.same_block_counterpart is True

    def test_no_txs_no_results(self) -> None:
        assert self.detector.detect([]) == []

    def test_two_txs_insufficient(self) -> None:
        prices = self._block_prices(45, 20)
        txs = [
            make_tx("0xf", 0, BOT_A, 45.0, all_gas_prices=prices),
            make_tx("0xv", 1, VICTIM, 20.0, all_gas_prices=prices),
        ]
        assert self.detector.detect(txs) == []
