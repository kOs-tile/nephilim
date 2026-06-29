"""
nephilim.classifier.jit_detector
----------------------------------
Just-In-Time (JIT) liquidity detection.

Algorithm
---------
JIT liquidity is a strategy where an MEV bot:
  1. Adds liquidity to a Uniswap V3 pool (``mint``) immediately before a
     large swap.
  2. Allows the swap to execute, earning fees on the position.
  3. Removes the liquidity (``decreaseLiquidity`` + ``collect``) immediately
     after the swap.

All three steps occur within the same block, making the liquidity provider
bear zero impermanent loss while collecting swap fees from the victim.

Detection criteria:
- There exists a ``mint`` (is_lp_add) and a ``decrease_liquidity``/``collect``
  (is_lp_remove) from the same address in the same block.
- A swap from a different address occurs between the add and remove positions.
- The pool address (to_address) is the same for all three transactions.

This implementation uses position-in-block ordering as a proxy for execution
order. Full confirmation requires EVM trace-level data.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from nephilim.stream.transaction_decoder import TxRecord

JITResult = Dict[str, Any]


class JITDetector:
    """
    Detects JIT liquidity provision patterns within a single block.
    """

    # A swap must exist within this many positions after the mint
    _MAX_SWAP_DISTANCE = 30
    # The remove must come within this many positions after the swap
    _MAX_REMOVE_DISTANCE = 30

    def detect(self, txs: List[TxRecord]) -> List[JITResult]:
        """
        Scan ``txs`` for JIT liquidity triples
        (add_liquidity → swap → remove_liquidity).

        Returns list of dicts describing each detected JIT event.
        """
        results: List[JITResult] = []

        # Index by pool address
        pool_to_txs: Dict[str, List[TxRecord]] = defaultdict(list)
        for tx in txs:
            if tx.to_address and (tx.is_lp_add or tx.is_lp_remove or tx.is_swap):
                pool_to_txs[tx.to_address.lower()].append(tx)

        for pool, pool_txs in pool_to_txs.items():
            if len(pool_txs) < 3:
                continue
            events = self._scan_pool(pool, pool_txs)
            results.extend(events)

        if results:
            logger.info("JITDetector found {} event(s) in block", len(results))
        return results

    def _scan_pool(self, pool: str, pool_txs: List[TxRecord]) -> List[JITResult]:
        """Find JIT triples for a specific pool."""
        results: List[JITResult] = []

        # Find all mints from each address
        adds: List[TxRecord] = [tx for tx in pool_txs if tx.is_lp_add]
        swaps: List[TxRecord] = [tx for tx in pool_txs if tx.is_swap]
        removes: List[TxRecord] = [tx for tx in pool_txs if tx.is_lp_remove]

        for add_tx in adds:
            # Find swaps that come after this add
            victim_swaps = [
                s for s in swaps
                if s.position_in_block > add_tx.position_in_block
                and s.from_address != add_tx.from_address
                and s.position_in_block - add_tx.position_in_block
                <= self._MAX_SWAP_DISTANCE
            ]

            for victim_swap in victim_swaps:
                # Find remove from same address as add, after the swap
                matching_removes = [
                    r for r in removes
                    if r.from_address == add_tx.from_address
                    and r.position_in_block > victim_swap.position_in_block
                    and r.position_in_block - victim_swap.position_in_block
                    <= self._MAX_REMOVE_DISTANCE
                ]

                for remove_tx in matching_removes:
                    # Mark counterparts
                    add_tx.same_block_counterpart = True
                    victim_swap.same_block_counterpart = True
                    remove_tx.same_block_counterpart = True

                    add_tx.pattern_flags.append("jit_liquidity")
                    victim_swap.pattern_flags.append("jit_liquidity")
                    remove_tx.pattern_flags.append("jit_liquidity")

                    fee_captured_eth = _estimate_jit_fee(victim_swap)
                    gas_cost_eth = _gas_cost_eth(add_tx) + _gas_cost_eth(remove_tx)

                    results.append(
                        {
                            "jit_provider": add_tx.from_address,
                            "victim": victim_swap.from_address,
                            "pool": pool,
                            "tx_hashes": [add_tx.hash, victim_swap.hash, remove_tx.hash],
                            "add_position": add_tx.position_in_block,
                            "swap_position": victim_swap.position_in_block,
                            "remove_position": remove_tx.position_in_block,
                            "estimated_fee_eth": fee_captured_eth,
                            "gas_cost_eth": gas_cost_eth,
                            "extracted_value_eth": max(
                                0.0, fee_captured_eth - gas_cost_eth
                            ),
                        }
                    )
                    # One remove per (add, swap) pair
                    break

        return results


def _estimate_jit_fee(swap_tx: TxRecord) -> float:
    """Estimate fee earned by the JIT provider from the victim swap."""
    # Uniswap V3 typical fee tier 0.3% — 0.05% is most common for JIT targets
    return swap_tx.value_eth * 0.003


def _gas_cost_eth(tx: TxRecord) -> float:
    return (tx.gas_price_wei * tx.gas_used) / 1e18
