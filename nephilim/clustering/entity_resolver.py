"""
nephilim.clustering.entity_resolver
--------------------------------------
Louvain community detection on the NetworkX transaction graph.

Each community becomes a "cluster" of wallets that exhibit shared behaviour
(co-activity, shared funders, counterpart interactions). The resolver assigns
a ``BehavioralProfile`` to each cluster based on the distribution of MEV types
and activity patterns within the cluster.

Behavioral profiles
-------------------
- ``mev_bot``          : High-gas, high-frequency, multiple MEV types
- ``market_maker``     : Frequent LP add/remove, bi-directional large flows
- ``whale``            : Large ETH values, infrequent trades, organic
- ``retail``           : Low value, random gas, organic swaps
- ``protocol_treasury``: Receives many inbound edges, sends protocol fees

The Louvain algorithm is applied to the undirected projection of the DiGraph
(edges weighted by ETH value + frequency), producing stable community labels
that can be persisted to Neo4j.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import community as community_louvain  # python-louvain
import networkx as nx
from loguru import logger

from nephilim.clustering.graph_builder import GraphBuilder

# ── Thresholds for behavioral profile classification ─────────────────────────
_MEV_BOT_MIN_MEV_TYPES = 2           # distinct MEV types seen
_MEV_BOT_MIN_GAS_PERCENTILE = 70.0   # average gas percentile
_MARKET_MAKER_MIN_LP_RATIO = 0.3     # fraction of cluster txs that are LP ops
_WHALE_MIN_AVG_VALUE_ETH = 10.0      # average ETH per outgoing tx
_MIN_CLUSTER_SIZE = 2                # ignore singleton nodes


@dataclass
class WalletSummary:
    address: str
    tx_count_out: int
    tx_count_in: int
    total_value_eth: float
    last_seen_block: int
    mev_types: Set[str] = field(default_factory=set)


@dataclass
class ClusterResult:
    cluster_id: str
    members: List[WalletSummary]
    behavioral_profile: str
    total_mev_extracted_eth: float
    dominant_mev_type: Optional[str]
    all_mev_types: Set[str]
    community_index: int


class EntityResolver:
    """
    Runs Louvain community detection on the current graph and returns
    ``ClusterResult`` objects for persisting to Neo4j.
    """

    def __init__(self, graph_builder: GraphBuilder) -> None:
        self._graph_builder = graph_builder
        self._last_partition: Dict[str, int] = {}

    def resolve(self, resolution: float = 1.0) -> List[ClusterResult]:
        """
        Run Louvain on the current transaction graph.

        Parameters
        ----------
        resolution:
            Louvain resolution parameter — higher values produce more, smaller
            communities.

        Returns
        -------
        List of ClusterResult objects, one per detected community.
        """
        graph = self._graph_builder.graph

        if graph.number_of_nodes() < 2:
            logger.debug("Graph too small for clustering ({} nodes)", graph.number_of_nodes())
            return []

        # Project to undirected, preserving edge weights
        undirected = graph.to_undirected()
        # Sum parallel edge weights that arise from the conversion
        for u, v, data in undirected.edges(data=True):
            if graph.has_edge(u, v) and graph.has_edge(v, u):
                data["weight"] = (
                    graph[u][v].get("weight", 0.0)
                    + graph[v][u].get("weight", 0.0)
                )

        # Remove very low weight edges to reduce noise
        edges_to_remove = [
            (u, v)
            for u, v, d in undirected.edges(data=True)
            if d.get("weight", 0.0) < 0.001
        ]
        undirected.remove_edges_from(edges_to_remove)

        if undirected.number_of_edges() == 0:
            logger.debug("No significant edges for clustering")
            return []

        # Run Louvain
        partition: Dict[str, int] = community_louvain.best_partition(
            undirected, weight="weight", resolution=resolution
        )
        self._last_partition = partition

        # Group addresses by community
        community_map: Dict[int, List[str]] = {}
        for address, community_id in partition.items():
            community_map.setdefault(community_id, []).append(address)

        results: List[ClusterResult] = []
        for community_id, members_addrs in community_map.items():
            if len(members_addrs) < _MIN_CLUSTER_SIZE:
                continue

            members = self._build_wallet_summaries(members_addrs, graph)
            profile, dominant_type = self._classify_profile(members, graph)
            all_mev_types = _collect_mev_types(members_addrs, graph)
            total_value = sum(m.total_value_eth for m in members)

            results.append(
                ClusterResult(
                    cluster_id=f"cluster-{community_id}",
                    members=members,
                    behavioral_profile=profile,
                    total_mev_extracted_eth=total_value,
                    dominant_mev_type=dominant_type,
                    all_mev_types=all_mev_types,
                    community_index=community_id,
                )
            )

        logger.info(
            "Louvain resolved {} communities from {} nodes",
            len(results),
            graph.number_of_nodes(),
        )
        return results

    # ──────────────────────────────────────────────────────────────────────────

    def _build_wallet_summaries(
        self, addresses: List[str], graph: nx.DiGraph
    ) -> List[WalletSummary]:
        summaries = []
        for addr in addresses:
            node_data = graph.nodes.get(addr, {})
            summaries.append(
                WalletSummary(
                    address=addr,
                    tx_count_out=node_data.get("tx_count_out", 0),
                    tx_count_in=node_data.get("tx_count_in", 0),
                    total_value_eth=node_data.get("total_value_eth", 0.0),
                    last_seen_block=node_data.get("last_seen_block", 0),
                    mev_types=node_data.get("mev_types", set()),
                )
            )
        return summaries

    def _classify_profile(
        self, members: List[WalletSummary], graph: nx.DiGraph
    ) -> tuple[str, Optional[str]]:
        """
        Assign a behavioral profile to a cluster based on its activity.
        """
        all_mev: Dict[str, int] = {}
        total_out = sum(m.tx_count_out for m in members)
        total_value = sum(m.total_value_eth for m in members)

        for m in members:
            for mev_type in m.mev_types:
                if mev_type not in ("unknown", "organic_swap", "shared_funder"):
                    all_mev[mev_type] = all_mev.get(mev_type, 0) + 1

        dominant_type = max(all_mev, key=all_mev.get) if all_mev else None
        distinct_mev_types = len(all_mev)

        avg_value = total_value / max(total_out, 1)

        # Protocol treasury: many inbound, few outbound
        in_ratio = sum(m.tx_count_in for m in members) / max(total_out, 1)
        if in_ratio > 5 and total_out < 10:
            return "protocol_treasury", dominant_type

        # MEV bot: multiple MEV types, high activity
        if distinct_mev_types >= _MEV_BOT_MIN_MEV_TYPES and total_out >= 10:
            return "mev_bot", dominant_type

        # Market maker: heavy LP activity
        lp_types = {
            "jit_liquidity", "cex_dex_arb", "pure_arb"
        }
        if dominant_type in lp_types:
            return "market_maker", dominant_type

        # Whale: large average value
        if avg_value >= _WHALE_MIN_AVG_VALUE_ETH:
            return "whale", dominant_type

        # Retail: everything else
        return "retail", dominant_type


def _collect_mev_types(
    addresses: List[str], graph: nx.DiGraph
) -> Set[str]:
    types: Set[str] = set()
    for addr in addresses:
        node_data = graph.nodes.get(addr, {})
        types.update(node_data.get("mev_types", set()))
    types.discard("unknown")
    return types
