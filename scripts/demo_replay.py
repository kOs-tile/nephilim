#!/usr/bin/env python3
"""
scripts/demo_replay.py
-----------------------
Replay a set of historically-representative synthetic blocks through the
full NEPHILIM MEV detection pipeline and print a rich formatted report.

No live node connection is required — this script builds mock block data
that encodes known MEV patterns from public Ethereum history.

The replay includes:
- Block 19_462_101 (representative sandwich attack on WETH/USDC pool)
- Block 19_462_102 (JIT liquidity provision event)
- Block 19_462_103 (CEX-DEX arbitrage burst, 3 bots)
- Block 19_462_104 (Flashloan + oracle manipulation attack)

Usage
-----
    python scripts/demo_replay.py
    python scripts/demo_replay.py --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from loguru import logger

# Make sure the package root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nephilim.classifier.mev_classifier import MEVClassifier
from nephilim.classifier.sandwich_detector import SandwichDetector
from nephilim.classifier.jit_detector import JITDetector
from nephilim.clustering.graph_builder import GraphBuilder
from nephilim.clustering.entity_resolver import EntityResolver
from nephilim.stream.transaction_decoder import TxRecord

console = Console()

# ── Known MEV actor addresses (anonymised) ────────────────────────────────────
BOT_A = "0xaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAa"
BOT_B = "0xbBbBbBbBbBbBbBbBbBbBbBbBbBbBbBbBbBbBbBb"
BOT_C = "0xcCcCcCcCcCcCcCcCcCcCcCcCcCcCcCcCcCcCcCc"
VICTIM_1 = "0x1111111111111111111111111111111111111111"
VICTIM_2 = "0x2222222222222222222222222222222222222222"
UNISWAP_V3 = "0xe592427a0aece92de3edee1f18e0157c05861564"
AAVE_POOL = "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9"

_BASE_FEE = int(20e9)  # 20 gwei


def _make_tx(
    hash: str,
    block_number: int,
    position: int,
    from_addr: str,
    to_addr: str,
    value_eth: float = 0.0,
    gas_price_gwei: float = 25.0,
    gas_used: int = 150_000,
    method_selector: str = "0x414bf389",
    is_swap: bool = True,
    is_lp_add: bool = False,
    is_lp_remove: bool = False,
    involves_flashloan: bool = False,
    touches_oracle: bool = False,
    is_governance: bool = False,
    block_gas_prices: List[int] | None = None,
) -> TxRecord:
    gas_price_wei = int(gas_price_gwei * 1e9)
    all_prices = block_gas_prices or [gas_price_wei]
    below = sum(1 for g in all_prices if g <= gas_price_wei)
    percentile = 100.0 * below / len(all_prices)

    premium = gas_price_wei / _BASE_FEE if _BASE_FEE else 1.0

    return TxRecord(
        hash=hash,
        block_number=block_number,
        position_in_block=position,
        from_address=from_addr,
        to_address=to_addr,
        value_wei=int(value_eth * 1e18),
        value_eth=value_eth,
        value_usd=value_eth * 3500.0,
        gas_price_wei=gas_price_wei,
        gas_used=gas_used,
        input_data="0x414bf389" + "0" * 200,
        method_selector=method_selector,
        is_swap=is_swap,
        is_lp_add=is_lp_add,
        is_lp_remove=is_lp_remove,
        involves_uniswap_v3=to_addr == UNISWAP_V3,
        involves_flashloan=involves_flashloan,
        touches_price_oracle=touches_oracle,
        is_governance=is_governance,
        gas_price_percentile=percentile,
        block_base_fee_wei=_BASE_FEE,
        gas_premium_multiplier=premium,
        priority_fee_wei=max(0, gas_price_wei - _BASE_FEE),
    )


def build_demo_blocks() -> List[Dict[str, Any]]:
    """Build 4 representative blocks with embedded MEV patterns."""

    # ── Block 1: Sandwich attack ─────────────────────────────────────────────
    prices_b1 = [int(g * 1e9) for g in [15, 18, 21, 22, 45, 45, 20, 19, 16, 17]]
    b1_txs = [
        _make_tx("0xfront001", 19_462_101, 3, BOT_A, UNISWAP_V3, 5.0, 45.0,
                 block_gas_prices=prices_b1),
        _make_tx("0xvictim01", 19_462_101, 4, VICTIM_1, UNISWAP_V3, 20.0, 21.0,
                 block_gas_prices=prices_b1),
        _make_tx("0xback001", 19_462_101, 5, BOT_A, UNISWAP_V3, 5.0, 44.0,
                 block_gas_prices=prices_b1),
        # Organic background txs
        _make_tx("0xorganic1", 19_462_101, 10, VICTIM_2, UNISWAP_V3, 0.5, 18.0,
                 block_gas_prices=prices_b1),
        _make_tx("0xorganic2", 19_462_101, 15, "0xabc0001", UNISWAP_V3, 0.3, 17.0,
                 block_gas_prices=prices_b1),
    ]

    # ── Block 2: JIT liquidity ───────────────────────────────────────────────
    prices_b2 = [int(g * 1e9) for g in [18, 22, 55, 55, 20, 19]]
    pool_addr = "0xpool88888888888888888888888888888888888"
    b2_txs = [
        _make_tx("0xjitadd1", 19_462_102, 2, BOT_B, pool_addr, 10.0, 55.0,
                 method_selector="0x88316456", is_swap=False, is_lp_add=True,
                 block_gas_prices=prices_b2),
        _make_tx("0xjitswap1", 19_462_102, 3, VICTIM_1, pool_addr, 15.0, 21.0,
                 block_gas_prices=prices_b2),
        _make_tx("0xjitrem1", 19_462_102, 4, BOT_B, pool_addr, 10.0, 54.0,
                 method_selector="0x0c49ccbe", is_swap=False, is_lp_remove=True,
                 block_gas_prices=prices_b2),
    ]

    # ── Block 3: CEX-DEX arb burst ───────────────────────────────────────────
    prices_b3 = [int(g * 1e9) for g in [85, 82, 80, 20, 19, 18, 17]]
    b3_txs = [
        _make_tx("0xarb001", 19_462_103, 0, BOT_A, UNISWAP_V3, 3.0, 85.0,
                 block_gas_prices=prices_b3),
        _make_tx("0xarb002", 19_462_103, 1, BOT_B, UNISWAP_V3, 2.5, 82.0,
                 block_gas_prices=prices_b3),
        _make_tx("0xarb003", 19_462_103, 2, BOT_C, UNISWAP_V3, 2.0, 80.0,
                 block_gas_prices=prices_b3),
        _make_tx("0xorgb001", 19_462_103, 10, VICTIM_2, UNISWAP_V3, 0.2, 19.0,
                 block_gas_prices=prices_b3),
    ]

    # ── Block 4: Flashloan + oracle manipulation ─────────────────────────────
    prices_b4 = [int(g * 1e9) for g in [95, 22, 19, 18, 20]]
    b4_txs = [
        _make_tx("0xflash001", 19_462_104, 0, BOT_A, AAVE_POOL, 500.0, 95.0,
                 method_selector="0xab9c4b5d", is_swap=False, is_lp_add=False,
                 involves_flashloan=True, touches_oracle=True,
                 gas_used=500_000, block_gas_prices=prices_b4),
        _make_tx("0xorgc001", 19_462_104, 5, VICTIM_2, UNISWAP_V3, 0.5, 20.0,
                 block_gas_prices=prices_b4),
    ]

    return [
        {"block_number": 19_462_101, "label": "Sandwich Attack", "txs": b1_txs},
        {"block_number": 19_462_102, "label": "JIT Liquidity", "txs": b2_txs},
        {"block_number": 19_462_103, "label": "CEX-DEX Arb Burst", "txs": b3_txs},
        {"block_number": 19_462_104, "label": "Flashloan + Oracle Manip", "txs": b4_txs},
    ]


def run_replay(verbose: bool = False) -> None:
    sandwich_detector = SandwichDetector()
    jit_detector = JITDetector()
    mev_classifier = MEVClassifier()  # will use rule-based fallback if no model
    graph_builder = GraphBuilder()
    entity_resolver = EntityResolver(graph_builder=graph_builder)

    blocks = build_demo_blocks()

    console.print(
        Panel.fit(
            "[bold cyan]NEPHILIM[/bold cyan] — Demo Block Replay\n"
            "[dim]Replaying 4 representative Ethereum blocks through the full pipeline[/dim]",
            border_style="cyan",
        )
    )

    all_events = []

    for block_data in blocks:
        block_number = block_data["block_number"]
        txs: List[TxRecord] = block_data["txs"]
        label = block_data["label"]

        console.print(f"\n[bold]Block #{block_number:,}[/bold] — [yellow]{label}[/yellow]")

        # Pattern detection
        sandwiches = sandwich_detector.detect(txs)
        jit_events = jit_detector.detect(txs)

        # Mark flags
        sandwich_hashes = {h for a in sandwiches for h in a["tx_hashes"]}
        jit_hashes = {h for e in jit_events for h in e["tx_hashes"]}
        for tx in txs:
            if tx.hash in sandwich_hashes:
                tx.pattern_flags.append("sandwich")
            if tx.hash in jit_hashes:
                tx.pattern_flags.append("jit_liquidity")

        # Classify
        classified = mev_classifier.predict_batch(txs)

        # Update graph
        for tx in classified:
            graph_builder.add_transaction(tx)

        # Print per-tx classification
        if verbose:
            tx_table = Table(
                "Hash", "From", "Type", "Confidence", "ETH", "Gas%ile",
                title=f"Block {block_number} Transactions",
            )
            for tx in classified:
                tx_table.add_row(
                    tx.hash[:12] + "…",
                    tx.from_address[:10] + "…",
                    tx.mev_type or "—",
                    f"{tx.mev_confidence:.2%}",
                    f"{tx.value_eth:.3f}",
                    f"{tx.gas_price_percentile:.0f}",
                )
            console.print(tx_table)

        # Print detected attacks
        for attack in sandwiches:
            console.print(
                f"  [red]🥪 Sandwich detected![/red] "
                f"Attacker: {attack['attacker'][:12]}… | "
                f"Extracted: {attack['extracted_value_eth']:.6f} ETH"
            )
            all_events.append(("sandwich_attack", block_number, attack["extracted_value_eth"]))

        for event in jit_events:
            console.print(
                f"  [yellow]⚡ JIT detected![/yellow] "
                f"Provider: {event['jit_provider'][:12]}… | "
                f"Extracted: {event['extracted_value_eth']:.6f} ETH"
            )
            all_events.append(("jit_liquidity", block_number, event["extracted_value_eth"]))

        mev_txs = [tx for tx in classified if tx.mev_type not in ("organic_swap", None)]
        if mev_txs and not sandwiches and not jit_events:
            for tx in mev_txs:
                console.print(
                    f"  [magenta]⚠ {tx.mev_type}[/magenta] "
                    f"by {tx.from_address[:12]}… "
                    f"(conf: {tx.mev_confidence:.2%})"
                )
                all_events.append((tx.mev_type, block_number, tx.value_eth))

    # Louvain clustering
    console.print("\n[bold]Running Louvain entity clustering…[/bold]")
    graph_builder.add_shared_funder_edges()
    clusters = entity_resolver.resolve()

    cluster_table = Table(
        "Cluster ID", "Profile", "Members", "MEV Types",
        title="Detected Entities",
    )
    for cl in clusters:
        cluster_table.add_row(
            cl.cluster_id,
            cl.behavioral_profile,
            str(len(cl.members)),
            ", ".join(sorted(cl.all_mev_types)[:3]) or "—",
        )
    console.print(cluster_table)

    # Summary
    console.print("\n")
    summary = Table("MEV Type", "Block", "Extracted ETH", title="Event Summary")
    for mev_type, block, eth in all_events:
        summary.add_row(mev_type, str(block), f"{eth:.6f}")
    console.print(summary)

    console.print(
        f"\n[bold green]Replay complete.[/bold green] "
        f"{len(all_events)} MEV events detected across {len(blocks)} blocks. "
        f"{len(clusters)} entity clusters resolved."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEPHILIM demo block replay")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show per-transaction table"
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="WARNING")  # suppress info for cleaner demo output

    run_replay(verbose=args.verbose)
