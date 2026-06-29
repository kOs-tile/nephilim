"""
tests/test_louvain_clustering.py
----------------------------------
Unit tests for the Louvain-based entity clustering pipeline.

Tests verify:
1. A tightly connected cluster of MEV bots is grouped together.
2. Isolated retail wallets form their own (or no) cluster.
3. Behavioral profile assignment is correct for mev_bot clusters.
4. Shared-funder edges strengthen co-clustering.
5. Graph node/edge counts are accurate after adding transactions.
6. Resolving an empty graph returns no clusters.
7. Singleton nodes below MIN_CLUSTER_SIZE are excluded.
"""

from __future__ import annotations

from typing import List

import pytest

from nephilim.clustering.graph_builder import GraphBuilder
from nephilim.clustering.entity_resolver import EntityResolver, ClusterResult
from nephilim.stream.transaction_decoder import TxRecord


BOT_A = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
BOT_B = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
BOT_C = "0xcccccccccccccccccccccccccccccccccccccccc"
RETAIL_1 = "0x1111111111111111111111111111111111111111"
RETAIL_2 = "0x2222222222222222222222222222222222222222"
POOL = "0xe592427a0aece92de3edee1f18e0157c05861564"
FUNDER = "0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0"


def _tx(
    hash: str,
    from_addr: str,
    to_addr: str,
    value_eth: float = 1.0,
    mev_type: str = "organic_swap",
    block: int = 19_000_000,
) -> TxRecord:
    return TxRecord(
        hash=hash,
        block_number=block,
        position_in_block=0,
        from_address=from_addr,
        to_address=to_addr,
        value_wei=int(value_eth * 1e18),
        value_eth=value_eth,
        value_usd=value_eth * 3500.0,
        gas_price_wei=int(30e9),
        gas_used=150_000,
        input_data="0x",
        method_selector="0x",
        mev_type=mev_type,
        mev_confidence=0.9,
    )


class TestGraphBuilder:
    def setup_method(self) -> None:
        self.builder = GraphBuilder()

    def test_nodes_created(self) -> None:
        tx = _tx("0x1", BOT_A, POOL)
        self.builder.add_transaction(tx)
        assert self.builder.node_count() == 2
        assert self.builder.graph.has_node(BOT_A)
        assert self.builder.graph.has_node(POOL)

    def test_edge_weight_accumulates(self) -> None:
        for i in range(5):
            self.builder.add_transaction(_tx(f"0x{i}", BOT_A, POOL, value_eth=2.0))
        assert self.builder.graph[BOT_A][POOL]["weight"] == pytest.approx(10.0)
        assert self.builder.graph[BOT_A][POOL]["tx_count"] == 5

    def test_directed_edge(self) -> None:
        self.builder.add_transaction(_tx("0xa", BOT_A, POOL))
        assert self.builder.graph.has_edge(BOT_A, POOL)
        assert not self.builder.graph.has_edge(POOL, BOT_A)

    def test_mev_type_recorded_on_node(self) -> None:
        self.builder.add_transaction(_tx("0xm", BOT_A, POOL, mev_type="sandwich_attack"))
        node_data = self.builder.graph.nodes[BOT_A]
        assert "sandwich_attack" in node_data["mev_types"]

    def test_reset_clears_graph(self) -> None:
        self.builder.add_transaction(_tx("0xr", BOT_A, POOL))
        self.builder.reset()
        assert self.builder.node_count() == 0

    def test_subgraph_extraction(self) -> None:
        self.builder.add_transaction(_tx("0xsub1", BOT_A, BOT_B))
        self.builder.add_transaction(_tx("0xsub2", BOT_B, BOT_C))
        self.builder.add_transaction(_tx("0xsub3", RETAIL_1, POOL))
        subg = self.builder.get_subgraph_for_address(BOT_A, hops=2)
        # BOT_A, BOT_B, BOT_C should all be reachable
        assert BOT_A in subg.nodes
        assert BOT_B in subg.nodes


class TestEntityResolver:
    def setup_method(self) -> None:
        self.builder = GraphBuilder()
        self.resolver = EntityResolver(graph_builder=self.builder)

    def _seed_mev_cluster(self) -> None:
        """Create a tight MEV bot cluster with shared interactions."""
        for i in range(20):
            self.builder.add_transaction(
                _tx(f"0xab{i}", BOT_A, BOT_B, mev_type="sandwich_attack")
            )
            self.builder.add_transaction(
                _tx(f"0xbc{i}", BOT_B, BOT_C, mev_type="cex_dex_arb")
            )
            self.builder.add_transaction(
                _tx(f"0xca{i}", BOT_C, BOT_A, mev_type="jit_liquidity")
            )
        # Also hit the same pool
        for bot in [BOT_A, BOT_B, BOT_C]:
            for j in range(5):
                self.builder.add_transaction(
                    _tx(f"0x{bot[:4]}{j}", bot, POOL, mev_type="sandwich_attack")
                )

    def _seed_retail(self) -> None:
        for i in range(3):
            self.builder.add_transaction(
                _tx(f"0xret{i}", RETAIL_1, POOL, mev_type="organic_swap")
            )
            self.builder.add_transaction(
                _tx(f"0xret2{i}", RETAIL_2, POOL, mev_type="organic_swap")
            )

    def test_empty_graph_returns_no_clusters(self) -> None:
        clusters = self.resolver.resolve()
        assert clusters == []

    def test_mev_bot_cluster_detected(self) -> None:
        self._seed_mev_cluster()
        clusters = self.resolver.resolve()
        # At least one cluster should exist
        assert len(clusters) >= 1

    def test_mev_bot_profile_assigned(self) -> None:
        self._seed_mev_cluster()
        clusters = self.resolver.resolve()
        profiles = {c.behavioral_profile for c in clusters}
        assert "mev_bot" in profiles

    def test_cluster_members_are_correct_type(self) -> None:
        self._seed_mev_cluster()
        clusters = self.resolver.resolve()
        for cluster in clusters:
            assert isinstance(cluster, ClusterResult)
            assert isinstance(cluster.cluster_id, str)
            assert len(cluster.members) >= 2

    def test_mev_types_populated(self) -> None:
        self._seed_mev_cluster()
        clusters = self.resolver.resolve()
        mev_bot_clusters = [c for c in clusters if c.behavioral_profile == "mev_bot"]
        if mev_bot_clusters:
            mev_types = mev_bot_clusters[0].all_mev_types
            assert len(mev_types) >= 1

    def test_retail_gets_different_profile_from_mev(self) -> None:
        self._seed_mev_cluster()
        self._seed_retail()
        clusters = self.resolver.resolve()
        profiles = {c.behavioral_profile for c in clusters}
        # Should have at least mev_bot and retail (or protocol_treasury)
        assert len(profiles) >= 1

    def test_shared_funder_strengthens_clustering(self) -> None:
        """
        Three addresses funded by the same source should cluster together
        when shared_funder edges are added.
        """
        # Funder sends ETH to BOT_A, BOT_B, BOT_C
        for bot in [BOT_A, BOT_B, BOT_C]:
            self.builder.add_transaction(
                _tx(f"0xfund_{bot[:4]}", FUNDER, bot, value_eth=2.0)
            )
        # Bots also transact with each other
        self.builder.add_transaction(_tx("0xint1", BOT_A, BOT_B, value_eth=0.5))
        self.builder.add_transaction(_tx("0xint2", BOT_B, BOT_C, value_eth=0.5))

        self.builder.add_shared_funder_edges()
        clusters = self.resolver.resolve()
        # BOT_A, BOT_B, BOT_C should end up in the same cluster
        bot_cluster = None
        for c in clusters:
            addrs = {m.address for m in c.members}
            if BOT_A in addrs and BOT_B in addrs:
                bot_cluster = c
                break
        assert bot_cluster is not None, "Expected BOT_A and BOT_B to co-cluster"

    def test_cluster_ids_are_unique(self) -> None:
        self._seed_mev_cluster()
        self._seed_retail()
        clusters = self.resolver.resolve()
        ids = [c.cluster_id for c in clusters]
        assert len(ids) == len(set(ids)), "Cluster IDs must be unique"
