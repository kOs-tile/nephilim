"""
nephilim.clustering.graph_builder
-----------------------------------
Builds and maintains a directed multi-signal NetworkX graph where:
- Nodes are Ethereum addresses.
- Edges represent transactions (from_address → to_address).
- Edge weight = cumulative ETH value transferred + frequency bonus.

Multi-signal edges
------------------
Each edge carries attributes:
- ``weight``          : cumulative ETH value (increases with each tx)
- ``tx_count``        : number of transactions on this edge
- ``last_block``      : most recent block number seen
- ``first_block``     : first block number seen
- ``mev_types``       : set of MEV types observed on this edge

Node attributes
---------------
- ``tx_count_out``    : outgoing transactions
- ``tx_count_in``     : incoming transactions
- ``total_value_eth`` : total ETH transferred out
- ``mev_types``       : set of MEV types
- ``last_seen_block`` : most recent block

Shared funding edges
--------------------
When two addresses both receive ETH from the same funder within a window,
an undirected ``shared_funder`` edge is added with weight 0.5. This helps
the Louvain algorithm cluster funded bots together.
"""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

import networkx as nx
from loguru import logger

from nephilim.stream.transaction_decoder import TxRecord

# How many recent addresses to track per funder for shared-funding detection
_FUNDER_WINDOW = 20
# Minimum ETH value for an edge to be included in clustering
_MIN_EDGE_WEIGHT = 0.001


class GraphBuilder:
    """
    Incrementally builds a NetworkX DiGraph from decoded transactions.
    Thread-unsafe: call from a single asyncio task.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        # funder_address → list of (recipient, block_number)
        self._funder_recipients: DefaultDict[
            str, List[Tuple[str, int]]
        ] = defaultdict(list)

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    def add_transaction(self, tx: TxRecord) -> None:
        """Incorporate a single decoded transaction into the graph."""
        if not tx.to_address:
            return

        src = tx.from_address
        dst = tx.to_address
        mev_type = tx.mev_type or "unknown"

        # Ensure nodes exist
        self._init_node(src)
        self._init_node(dst)

        # Update node attributes
        self._graph.nodes[src]["tx_count_out"] = (
            self._graph.nodes[src].get("tx_count_out", 0) + 1
        )
        self._graph.nodes[src]["total_value_eth"] = (
            self._graph.nodes[src].get("total_value_eth", 0.0) + tx.value_eth
        )
        self._graph.nodes[src]["last_seen_block"] = max(
            self._graph.nodes[src].get("last_seen_block", 0), tx.block_number
        )
        self._graph.nodes[src].setdefault("mev_types", set()).add(mev_type)

        self._graph.nodes[dst]["tx_count_in"] = (
            self._graph.nodes[dst].get("tx_count_in", 0) + 1
        )
        self._graph.nodes[dst]["last_seen_block"] = max(
            self._graph.nodes[dst].get("last_seen_block", 0), tx.block_number
        )

        # Update / create directed edge
        if self._graph.has_edge(src, dst):
            self._graph[src][dst]["weight"] += tx.value_eth
            self._graph[src][dst]["tx_count"] += 1
            self._graph[src][dst]["last_block"] = tx.block_number
            self._graph[src][dst].setdefault("mev_types", set()).add(mev_type)
        else:
            self._graph.add_edge(
                src,
                dst,
                weight=tx.value_eth,
                tx_count=1,
                first_block=tx.block_number,
                last_block=tx.block_number,
                mev_types={mev_type},
            )

        # Track for shared-funder detection (only high-value ETH transfers)
        if tx.value_eth >= 0.1:
            self._update_funder_index(src, dst, tx.block_number)

    def add_shared_funder_edges(self) -> None:
        """
        Scan the funder index and add undirected ``shared_funder`` edges
        between co-funded addresses to strengthen clustering.
        """
        edges_added = 0
        for funder, recipients in self._funder_recipients.items():
            addrs = [r for r, _ in recipients[-_FUNDER_WINDOW:]]
            if len(addrs) < 2:
                continue
            for i, a in enumerate(addrs):
                for b in addrs[i + 1:]:
                    if not self._graph.has_edge(a, b) and not self._graph.has_edge(b, a):
                        self._graph.add_edge(
                            a, b,
                            weight=0.5,
                            tx_count=0,
                            first_block=0,
                            last_block=0,
                            mev_types={"shared_funder"},
                            edge_type="shared_funder",
                        )
                        edges_added += 1
        if edges_added:
            logger.debug("Added {} shared-funder edges", edges_added)

    def get_subgraph_for_address(
        self, address: str, hops: int = 2
    ) -> nx.DiGraph:
        """Return the ego network (within ``hops``) centred on ``address``."""
        ego = nx.ego_graph(self._graph, address, radius=hops)
        return ego

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def reset(self) -> None:
        """Clear the graph (useful for tests)."""
        self._graph.clear()
        self._funder_recipients.clear()

    # ──────────────────────────────────────────────────────────────────────────

    def _init_node(self, address: str) -> None:
        if not self._graph.has_node(address):
            self._graph.add_node(
                address,
                tx_count_out=0,
                tx_count_in=0,
                total_value_eth=0.0,
                last_seen_block=0,
                mev_types=set(),
            )

    def _update_funder_index(
        self, funder: str, recipient: str, block: int
    ) -> None:
        records = self._funder_recipients[funder]
        records.append((recipient, block))
        # Trim to window size
        if len(records) > _FUNDER_WINDOW * 2:
            self._funder_recipients[funder] = records[-_FUNDER_WINDOW:]
