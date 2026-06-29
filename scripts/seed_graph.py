#!/usr/bin/env python3
"""
scripts/seed_graph.py
-----------------------
Seed Neo4j with a representative entity graph for demo and testing purposes.

Creates:
- 3 MEV bot wallets (cluster "mev-alpha")
- 1 market maker wallet
- 2 retail wallets
- 1 protocol treasury (Uniswap V3 router)
- Directed transaction edges between them
- MEMBER_OF relationships linking wallets to clusters

Usage
-----
    python scripts/seed_graph.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from loguru import logger
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nephilim.config import get_settings
from nephilim.storage.neo4j_client import Neo4jClient
from nephilim.clustering.entity_resolver import ClusterResult, WalletSummary

console = Console()

# ── Demo entity data ──────────────────────────────────────────────────────────

_MEV_BOT_1 = "0xaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAa"
_MEV_BOT_2 = "0xbBbBbBbBbBbBbBbBbBbBbBbBbBbBbBbBbBbBbBb"
_MEV_BOT_3 = "0xcCcCcCcCcCcCcCcCcCcCcCcCcCcCcCcCcCcCcCc"
_MARKET_MAKER = "0xdDdDdDdDdDdDdDdDdDdDdDdDdDdDdDdDdDdDdDd"
_RETAIL_1 = "0x1111111111111111111111111111111111111111"
_RETAIL_2 = "0x2222222222222222222222222222222222222222"
_UNISWAP_V3_ROUTER = "0xe592427a0aece92de3edee1f18e0157c05861564"
_FUNDER = "0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0"


async def seed(neo4j: Neo4jClient) -> None:
    await neo4j.ensure_constraints()

    console.print("[bold]Seeding wallets…[/bold]")

    # MEV bots
    for addr, label in [
        (_MEV_BOT_1, "MEV-Bot-Alpha-1"),
        (_MEV_BOT_2, "MEV-Bot-Alpha-2"),
        (_MEV_BOT_3, "MEV-Bot-Alpha-3"),
    ]:
        await neo4j.upsert_wallet(
            address=addr,
            label=label,
            behavioral_profile="mev_bot",
            tx_count_out=1250,
            tx_count_in=300,
            total_value_eth=88.5,
            last_seen_block=19_500_000,
        )

    # Market maker
    await neo4j.upsert_wallet(
        address=_MARKET_MAKER,
        label="MarketMaker-Gamma",
        behavioral_profile="market_maker",
        tx_count_out=450,
        tx_count_in=460,
        total_value_eth=342.1,
        last_seen_block=19_499_900,
    )

    # Retail users
    await neo4j.upsert_wallet(
        address=_RETAIL_1,
        label=None,
        behavioral_profile="retail",
        tx_count_out=12,
        tx_count_in=8,
        total_value_eth=1.4,
        last_seen_block=19_499_800,
    )
    await neo4j.upsert_wallet(
        address=_RETAIL_2,
        label=None,
        behavioral_profile="retail",
        tx_count_out=7,
        tx_count_in=5,
        total_value_eth=0.8,
        last_seen_block=19_499_750,
    )

    # Funder wallet
    await neo4j.upsert_wallet(
        address=_FUNDER,
        label="Common-Funder",
        behavioral_profile="whale",
        tx_count_out=3,
        tx_count_in=1,
        total_value_eth=5.0,
        last_seen_block=19_460_000,
    )

    # Protocol contract
    await neo4j.upsert_contract(_UNISWAP_V3_ROUTER)
    console.print("[green]✓[/green] Wallets and contracts seeded.")

    # ── Create clusters ───────────────────────────────────────────────────────
    console.print("[bold]Creating clusters…[/bold]")

    mev_cluster = ClusterResult(
        cluster_id="cluster-mev-alpha",
        members=[
            WalletSummary(
                address=addr,
                tx_count_out=1250,
                tx_count_in=300,
                total_value_eth=88.5,
                last_seen_block=19_500_000,
                mev_types={"sandwich_attack", "cex_dex_arb", "jit_liquidity"},
            )
            for addr in [_MEV_BOT_1, _MEV_BOT_2, _MEV_BOT_3]
        ],
        behavioral_profile="mev_bot",
        total_mev_extracted_eth=265.5,
        dominant_mev_type="sandwich_attack",
        all_mev_types={"sandwich_attack", "cex_dex_arb", "jit_liquidity"},
        community_index=0,
    )

    mm_cluster = ClusterResult(
        cluster_id="cluster-market-maker-gamma",
        members=[
            WalletSummary(
                address=_MARKET_MAKER,
                tx_count_out=450,
                tx_count_in=460,
                total_value_eth=342.1,
                last_seen_block=19_499_900,
                mev_types={"jit_liquidity", "pure_arb"},
            )
        ],
        behavioral_profile="market_maker",
        total_mev_extracted_eth=342.1,
        dominant_mev_type="jit_liquidity",
        all_mev_types={"jit_liquidity", "pure_arb"},
        community_index=1,
    )

    await neo4j.upsert_cluster(mev_cluster)
    await neo4j.upsert_cluster(mm_cluster)
    console.print("[green]✓[/green] Clusters created and wallets linked.")

    console.print(
        "\n[bold green]Seed complete.[/bold green] "
        "Neo4j now contains a demo entity graph. "
        "Query it at [cyan]http://localhost:7474[/cyan] or via the GraphQL API."
    )


async def main() -> None:
    settings = get_settings()
    neo4j = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    try:
        await seed(neo4j)
    finally:
        await neo4j.close()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    asyncio.run(main())
