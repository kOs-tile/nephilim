"""
nephilim.storage.neo4j_client
--------------------------------
Neo4j entity graph persistence layer.

Graph Schema
------------

Nodes:
  (:Wallet {address, label, behavioral_profile, first_seen_block, last_seen_block,
            tx_count_out, tx_count_in, total_value_eth})

  (:Cluster {cluster_id, behavioral_profile, member_count,
             total_mev_extracted_eth, dominant_mev_type, created_at})

  (:Contract {address, name, protocol, is_dex, is_lending, created_block})

Relationships:
  (:Wallet)-[:SENT_TX {block, value_eth, mev_type, confidence}]->(:Wallet)
  (:Wallet)-[:SENT_TX]->(:Contract)
  (:Wallet)-[:MEMBER_OF]->(:Cluster)
  (:Wallet)-[:INTERACTED_WITH {count, last_block, value_eth}]->(:Contract)

Constraints / Indexes:
  UNIQUE on Wallet.address, Cluster.cluster_id, Contract.address
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession

from nephilim.clustering.entity_resolver import ClusterResult
from nephilim.stream.transaction_decoder import TxRecord

# Known DEX/protocol contract labels
_CONTRACT_LABELS: Dict[str, Dict[str, Any]] = {
    "0xe592427a0aece92de3edee1f18e0157c05861564": {
        "name": "UniswapV3Router",
        "protocol": "Uniswap",
        "is_dex": True,
        "is_lending": False,
    },
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d": {
        "name": "UniswapV2Router02",
        "protocol": "Uniswap",
        "is_dex": True,
        "is_lending": False,
    },
    "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9": {
        "name": "AaveV2LendingPool",
        "protocol": "Aave",
        "is_dex": False,
        "is_lending": True,
    },
}


class Neo4jClient:
    """
    Async Neo4j client for the NEPHILIM entity graph.
    All write operations use parameterised Cypher to prevent injection.
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._driver: Optional[AsyncDriver] = None

    async def _get_driver(self) -> AsyncDriver:
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
                max_connection_lifetime=3600,
            )
        return self._driver

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            self._driver = None

    # ──────────────────────────────────────────────────────────────────────────
    # Schema initialisation
    # ──────────────────────────────────────────────────────────────────────────

    async def ensure_constraints(self) -> None:
        """Create uniqueness constraints and indexes if they don't exist."""
        driver = await self._get_driver()
        async with driver.session() as session:
            constraints = [
                "CREATE CONSTRAINT wallet_address_unique IF NOT EXISTS "
                "FOR (w:Wallet) REQUIRE w.address IS UNIQUE",
                "CREATE CONSTRAINT cluster_id_unique IF NOT EXISTS "
                "FOR (c:Cluster) REQUIRE c.cluster_id IS UNIQUE",
                "CREATE CONSTRAINT contract_address_unique IF NOT EXISTS "
                "FOR (c:Contract) REQUIRE c.address IS UNIQUE",
            ]
            indexes = [
                "CREATE INDEX wallet_profile_idx IF NOT EXISTS "
                "FOR (w:Wallet) ON (w.behavioral_profile)",
                "CREATE INDEX cluster_profile_idx IF NOT EXISTS "
                "FOR (c:Cluster) ON (c.behavioral_profile)",
            ]
            for cypher in constraints + indexes:
                try:
                    await session.run(cypher)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Constraint/index setup note: {}", exc)
        logger.info("Neo4j constraints and indexes verified.")

    # ──────────────────────────────────────────────────────────────────────────
    # Entity upserts
    # ──────────────────────────────────────────────────────────────────────────

    async def upsert_wallet(
        self,
        address: str,
        label: Optional[str] = None,
        behavioral_profile: Optional[str] = None,
        tx_count_out: int = 0,
        tx_count_in: int = 0,
        total_value_eth: float = 0.0,
        last_seen_block: int = 0,
    ) -> None:
        driver = await self._get_driver()
        async with driver.session() as session:
            await session.run(
                """
                MERGE (w:Wallet {address: $address})
                ON CREATE SET
                    w.label = $label,
                    w.behavioral_profile = $profile,
                    w.tx_count_out = $tx_out,
                    w.tx_count_in = $tx_in,
                    w.total_value_eth = $value,
                    w.first_seen_block = $block,
                    w.last_seen_block = $block,
                    w.created_at = $now
                ON MATCH SET
                    w.label = CASE WHEN $label IS NOT NULL THEN $label ELSE w.label END,
                    w.behavioral_profile = CASE WHEN $profile IS NOT NULL
                        THEN $profile ELSE w.behavioral_profile END,
                    w.tx_count_out = w.tx_count_out + $tx_out,
                    w.tx_count_in = w.tx_count_in + $tx_in,
                    w.total_value_eth = w.total_value_eth + $value,
                    w.last_seen_block = CASE WHEN $block > w.last_seen_block
                        THEN $block ELSE w.last_seen_block END,
                    w.updated_at = $now
                """,
                address=address.lower(),
                label=label,
                profile=behavioral_profile,
                tx_out=tx_count_out,
                tx_in=tx_count_in,
                value=total_value_eth,
                block=last_seen_block,
                now=datetime.now(timezone.utc).isoformat(),
            )

    async def upsert_contract(self, address: str) -> None:
        meta = _CONTRACT_LABELS.get(address.lower(), {})
        driver = await self._get_driver()
        async with driver.session() as session:
            await session.run(
                """
                MERGE (c:Contract {address: $address})
                ON CREATE SET
                    c.name = $name,
                    c.protocol = $protocol,
                    c.is_dex = $is_dex,
                    c.is_lending = $is_lending,
                    c.created_at = $now
                """,
                address=address.lower(),
                name=meta.get("name"),
                protocol=meta.get("protocol"),
                is_dex=meta.get("is_dex", False),
                is_lending=meta.get("is_lending", False),
                now=datetime.now(timezone.utc).isoformat(),
            )

    async def upsert_transaction_edge(self, tx: TxRecord) -> None:
        if not tx.to_address:
            return
        driver = await self._get_driver()
        async with driver.session() as session:
            await session.run(
                """
                MATCH (src:Wallet {address: $from_addr})
                MATCH (dst {address: $to_addr})
                MERGE (src)-[r:SENT_TX {hash: $hash}]->(dst)
                ON CREATE SET
                    r.block = $block,
                    r.value_eth = $value,
                    r.mev_type = $mev_type,
                    r.confidence = $confidence,
                    r.gas_price_gwei = $gas_gwei
                """,
                from_addr=tx.from_address,
                to_addr=tx.to_address,
                hash=tx.hash,
                block=tx.block_number,
                value=tx.value_eth,
                mev_type=tx.mev_type,
                confidence=tx.mev_confidence,
                gas_gwei=tx.gas_price_wei / 1e9,
            )

    async def upsert_cluster(self, cluster: ClusterResult) -> None:
        driver = await self._get_driver()
        async with driver.session() as session:
            # Create/update cluster node
            await session.run(
                """
                MERGE (c:Cluster {cluster_id: $cluster_id})
                ON CREATE SET
                    c.behavioral_profile = $profile,
                    c.member_count = $member_count,
                    c.total_mev_extracted_eth = $mev_eth,
                    c.dominant_mev_type = $dominant,
                    c.created_at = $now
                ON MATCH SET
                    c.behavioral_profile = $profile,
                    c.member_count = $member_count,
                    c.total_mev_extracted_eth = $mev_eth,
                    c.dominant_mev_type = $dominant,
                    c.updated_at = $now
                """,
                cluster_id=cluster.cluster_id,
                profile=cluster.behavioral_profile,
                member_count=len(cluster.members),
                mev_eth=cluster.total_mev_extracted_eth,
                dominant=cluster.dominant_mev_type,
                now=datetime.now(timezone.utc).isoformat(),
            )

            # Link members to cluster
            for member in cluster.members:
                await session.run(
                    """
                    MERGE (w:Wallet {address: $address})
                    WITH w
                    MATCH (c:Cluster {cluster_id: $cluster_id})
                    MERGE (w)-[:MEMBER_OF]->(c)
                    SET w.behavioral_profile = $profile
                    """,
                    address=member.address,
                    cluster_id=cluster.cluster_id,
                    profile=cluster.behavioral_profile,
                )

    async def upsert_entities(self, clusters: List[ClusterResult]) -> None:
        """Batch upsert all cluster results."""
        for cluster in clusters:
            await self.upsert_cluster(cluster)
        logger.debug("Neo4j: upserted {} clusters", len(clusters))

    # ──────────────────────────────────────────────────────────────────────────
    # Query helpers
    # ──────────────────────────────────────────────────────────────────────────

    async def get_wallet(self, address: str) -> Optional[Dict[str, Any]]:
        driver = await self._get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (w:Wallet {address: $address})
                OPTIONAL MATCH (w)-[:MEMBER_OF]->(c:Cluster)
                RETURN w, c
                """,
                address=address.lower(),
            )
            record = await result.single()
            if not record:
                return None
            wallet_data = dict(record["w"])
            cluster_data = dict(record["c"]) if record["c"] else None
            return {"wallet": wallet_data, "cluster": cluster_data}

    async def get_cluster(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        driver = await self._get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (c:Cluster {cluster_id: $cluster_id})
                OPTIONAL MATCH (w:Wallet)-[:MEMBER_OF]->(c)
                RETURN c, collect(w) AS members
                """,
                cluster_id=cluster_id,
            )
            record = await result.single()
            if not record:
                return None
            return {
                "cluster": dict(record["c"]),
                "members": [dict(m) for m in record["members"]],
            }

    async def who_touched_contract(
        self, contract_address: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        driver = await self._get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (w:Wallet)-[r:SENT_TX]->(c:Contract {address: $address})
                OPTIONAL MATCH (w)-[:MEMBER_OF]->(cl:Cluster)
                WITH w, cl, count(r) AS interaction_count,
                     max(r.block) AS last_block,
                     sum(r.value_eth) AS total_value
                RETURN w.address AS address,
                       interaction_count,
                       last_block,
                       total_value,
                       cl.cluster_id AS cluster_id,
                       cl.behavioral_profile AS cluster_profile
                ORDER BY interaction_count DESC
                LIMIT $limit
                """,
                address=contract_address.lower(),
                limit=limit,
            )
            return [dict(r) async for r in result]
