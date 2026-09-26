"""
nephilim.classifier.sandwich_detector
---------------------------------------
Block-level sandwich attack detector.

Algorithm
---------
A sandwich attack is a (frontrun → victim → backrun) triple where:
  1. All three transactions interact with the same DEX token pair.
  2. The frontrun tx (position i) has a higher gas price than the victim (position j).
  3. The backrun tx (position k) has a higher gas price than the victim and
     comes after both i and j.
  4. The frontrun and backrun originate from the same ``from_address``.
  5. The victim is a swap on the same pair.

Extracted value is estimated as the difference between the ETH values of
the backrun output and frontrun input (simplified heuristic; exact profit
requires trace-level data).

The detector is intentionally strict to minimise false positives: it requires
a decoded canonical DEX pair key plus token direction evidence. The frontrun and
victim must share a direction and the backrun must exactly reverse it. Router
address alone is never treated as a pair.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from nephilim.stream.transaction_decoder import TxRecord

# Swap method selectors shared with transaction_decoder
_SWAP_SELECTORS = {
    "0x7ff36ab5",  # UNI V2 swapExactETHForTokens
    "0x18cbafe5",  # UNI V2 swapExactTokensForETH
    "0x38ed1739",  # UNI V2 swapExactTokensForTokens
    "0xb6f9de95",  # UNI V2 swapETHForExactTokens
    "0x414bf389",  # UNI V3 exactInputSingle
    "0xc04b8d59",  # UNI V3 exactInput
    "0xdb3e2198",  # UNI V3 exactOutputSingle
}


def _is_swap(tx: TxRecord) -> bool:
    return tx.method_selector in _SWAP_SELECTORS


def _pair_key(tx: TxRecord) -> Optional[str]:
    """
    Return a decoded canonical token-pair key.

    Router address is intentionally not used as a fallback: grouping unrelated
    swaps by a shared router creates high-confidence-looking false positives.
    Unsupported calldata therefore fails closed.
    """
    return tx.dex_pair_key


def _has_sandwich_direction(
    front: TxRecord,
    victim: TxRecord,
    back: TxRecord,
) -> bool:
    """Require front/victim same direction and backrun exact reverse direction."""
    fields = (
        front.dex_token_in,
        front.dex_token_out,
        victim.dex_token_in,
        victim.dex_token_out,
        back.dex_token_in,
        back.dex_token_out,
    )
    if any(value is None for value in fields):
        return False

    return (
        front.dex_token_in == victim.dex_token_in
        and front.dex_token_out == victim.dex_token_out
        and back.dex_token_in == front.dex_token_out
        and back.dex_token_out == front.dex_token_in
    )


SandwichResult = Dict[str, Any]


class SandwichDetector:
    """Detects sandwich attacks within a single block's transaction list."""

    # Minimum gas price ratio between attacker and victim to consider
    _MIN_GAS_RATIO = 1.05
    # Maximum gap between frontrun index and backrun index
    _MAX_BLOCK_SPAN = 50

    def detect(self, txs: List[TxRecord]) -> List[SandwichResult]:
        """
        Scan ``txs`` (in block order) for sandwich triples.

        Returns a list of result dicts, each containing:
        - ``attacker``: from_address of the sandwicher
        - ``victim``: from_address of the victim
        - ``tx_hashes``: [frontrun_hash, victim_hash, backrun_hash]
        - ``pair``: decoded canonical DEX pair key
        - ``extracted_value_eth``: estimated extracted ETH
        - ``frontrun_position``: block index of frontrun
        - ``victim_position``: block index of victim
        - ``backrun_position``: block index of backrun
        """
        results: List[SandwichResult] = []

        # Index swap txs by pair key
        pair_to_txs: Dict[str, List[TxRecord]] = defaultdict(list)
        for tx in txs:
            if _is_swap(tx):
                key = _pair_key(tx)
                if key:
                    pair_to_txs[key].append(tx)

        for pair, pair_txs in pair_to_txs.items():
            if len(pair_txs) < 3:
                continue

            attacks = self._scan_pair(pair, pair_txs)
            results.extend(attacks)

        if results:
            logger.info(
                "SandwichDetector found {} attack(s) in block",
                len(results),
            )
        return results

    def _scan_pair(
        self, pair: str, pair_txs: List[TxRecord]
    ) -> List[SandwichResult]:
        """
        For a given pair, find all (frontrun, victim, backrun) triples.
        """
        results: List[SandwichResult] = []
        n = len(pair_txs)

        for fi in range(n):
            front = pair_txs[fi]

            for vi in range(fi + 1, min(fi + self._MAX_BLOCK_SPAN, n)):
                victim = pair_txs[vi]

                # Victim must be a different address
                if victim.from_address == front.from_address:
                    continue

                # Frontrun gas must be strictly higher than victim
                if front.gas_price_wei < victim.gas_price_wei * self._MIN_GAS_RATIO:
                    continue

                # Look for matching backrun
                for bi in range(vi + 1, min(vi + self._MAX_BLOCK_SPAN, n)):
                    back = pair_txs[bi]

                    # Backrun must come from same address as frontrun
                    if back.from_address != front.from_address:
                        continue

                    # Backrun gas must also be higher than victim
                    if back.gas_price_wei < victim.gas_price_wei * self._MIN_GAS_RATIO:
                        continue

                    # A genuine sandwich requires directional evidence:
                    # front and victim move the same way through the pair,
                    # while the backrun reverses the attacker's frontrun.
                    if not _has_sandwich_direction(front, victim, back):
                        continue

                    # Confirm they span <= MAX_BLOCK_SPAN positions overall
                    span = back.position_in_block - front.position_in_block
                    if span > self._MAX_BLOCK_SPAN:
                        continue

                    # Mark counterparts
                    front.same_block_counterpart = True
                    back.same_block_counterpart = True
                    victim.same_block_counterpart = True

                    extracted = _estimate_extracted_value(front, victim, back)

                    results.append(
                        {
                            "attacker": front.from_address,
                            "victim": victim.from_address,
                            "pair": pair,
                            "tx_hashes": [front.hash, victim.hash, back.hash],
                            "frontrun_position": front.position_in_block,
                            "victim_position": victim.position_in_block,
                            "backrun_position": back.position_in_block,
                            "extracted_value_eth": extracted,
                            "extracted_value_is_estimate": True,
                            "estimate_method": "victim_value_fraction_minus_gas",
                            "evidence": {
                                "pair_decoded": True,
                                "direction_verified": True,
                                "front_direction": [
                                    front.dex_token_in,
                                    front.dex_token_out,
                                ],
                                "victim_direction": [
                                    victim.dex_token_in,
                                    victim.dex_token_out,
                                ],
                                "back_direction": [
                                    back.dex_token_in,
                                    back.dex_token_out,
                                ],
                                "gas_ratio_front_vs_victim": round(
                                    front.gas_price_wei / victim.gas_price_wei, 6
                                ) if victim.gas_price_wei else None,
                                "gas_ratio_back_vs_victim": round(
                                    back.gas_price_wei / victim.gas_price_wei, 6
                                ) if victim.gas_price_wei else None,
                            },
                            "frontrun_gas_price_gwei": front.gas_price_wei / 1e9,
                            "victim_gas_price_gwei": victim.gas_price_wei / 1e9,
                            "backrun_gas_price_gwei": back.gas_price_wei / 1e9,
                        }
                    )
                    # One backrun per (frontrun, victim) pair
                    break

        return results


def _estimate_extracted_value(
    frontrun: TxRecord, victim: TxRecord, backrun: TxRecord
) -> float:
    """
    Heuristic: extracted ETH ≈ gas cost paid × premium + victim value fraction.
    A precise calculation requires trace-level simulation; this is a fast proxy.
    """
    gas_cost_front = (frontrun.gas_price_wei * frontrun.gas_used) / 1e18
    gas_cost_back = (backrun.gas_price_wei * backrun.gas_used) / 1e18
    victim_value_fraction = victim.value_eth * 0.003  # ~0.3% AMM fee captured
    extracted = victim_value_fraction - gas_cost_front - gas_cost_back
    return max(0.0, round(extracted, 6))
