"""
nephilim.api.graphql_schema
------------------------------
Strawberry GraphQL schema for NEPHILIM.

Queries
-------
- ``wallet(address: String!)`` — look up a single wallet by address
- ``cluster(id: String!)`` — look up a cluster by ID
- ``mevSummary(fromBlock: Int!, toBlock: Int!)`` — MEV stats for a block range
- ``whoTouched(contract: String!, limit: Int)`` — wallets that called a contract
"""

from __future__ import annotations

from typing import List, Optional

import strawberry
from strawberry.types import Info
from loguru import logger


# ── GraphQL Types ────────────────────────────────────────────────────────────

@strawberry.type
class MEVActivity:
    mev_type: str
    count: int
    total_value_eth: float


@strawberry.type
class ClusterInfo:
    cluster_id: str
    behavioral_profile: str
    member_count: int
    total_mev_extracted_eth: float
    dominant_mev_type: Optional[str]


@strawberry.type
class WalletType:
    address: str
    label: Optional[str]
    behavioral_profile: Optional[str]
    tx_count_out: int
    tx_count_in: int
    total_value_eth: float
    last_seen_block: int
    cluster: Optional[ClusterInfo]
    mev_activity: List[MEVActivity]


@strawberry.type
class MEVByType:
    mev_type: str
    count: int
    extracted_eth: float


@strawberry.type
class TopActor:
    address: str
    extracted_eth: float


@strawberry.type
class MEVSummary:
    from_block: int
    to_block: int
    total_extracted_eth: float
    by_type: List[MEVByType]
    top_actors: List[TopActor]


@strawberry.type
class ContractInteractor:
    address: str
    interaction_count: int
    last_block: int
    total_value_eth: float
    cluster_id: Optional[str]
    cluster_profile: Optional[str]


@strawberry.type
class ClusterMember:
    address: str
    tx_count_out: int
    tx_count_in: int
    total_value_eth: float
    last_seen_block: int


@strawberry.type
class ClusterDetail:
    cluster_id: str
    behavioral_profile: str
    member_count: int
    total_mev_extracted_eth: float
    dominant_mev_type: Optional[str]
    members: List[ClusterMember]


# ── Resolvers ─────────────────────────────────────────────────────────────────

@strawberry.type
class Query:

    @strawberry.field(description="Look up a wallet by Ethereum address.")
    async def wallet(self, address: str, info: Info) -> Optional[WalletType]:
        neo4j = info.context["neo4j_client"]
        timescale = info.context["timescale_client"]

        result = await neo4j.get_wallet(address)
        if not result:
            logger.debug("GraphQL wallet({}) — not found", address)
            return None

        wallet_data = result["wallet"]
        cluster_data = result.get("cluster")

        mev_raw = await timescale.get_wallet_mev_activity(address)
        mev_activity = [
            MEVActivity(
                mev_type=r["mev_type"] or "unknown",
                count=int(r["count"]),
                total_value_eth=float(r["total_value_eth"] or 0),
            )
            for r in mev_raw
        ]

        cluster_info: Optional[ClusterInfo] = None
        if cluster_data:
            cluster_info = ClusterInfo(
                cluster_id=cluster_data.get("cluster_id", ""),
                behavioral_profile=cluster_data.get("behavioral_profile", "unknown"),
                member_count=int(cluster_data.get("member_count", 0)),
                total_mev_extracted_eth=float(
                    cluster_data.get("total_mev_extracted_eth", 0)
                ),
                dominant_mev_type=cluster_data.get("dominant_mev_type"),
            )

        return WalletType(
            address=wallet_data.get("address", address),
            label=wallet_data.get("label"),
            behavioral_profile=wallet_data.get("behavioral_profile"),
            tx_count_out=int(wallet_data.get("tx_count_out", 0)),
            tx_count_in=int(wallet_data.get("tx_count_in", 0)),
            total_value_eth=float(wallet_data.get("total_value_eth", 0)),
            last_seen_block=int(wallet_data.get("last_seen_block", 0)),
            cluster=cluster_info,
            mev_activity=mev_activity,
        )

    @strawberry.field(description="Look up a cluster by its ID.")
    async def cluster(self, id: str, info: Info) -> Optional[ClusterDetail]:
        neo4j = info.context["neo4j_client"]
        result = await neo4j.get_cluster(id)
        if not result:
            return None

        cluster_data = result["cluster"]
        members = [
            ClusterMember(
                address=m.get("address", ""),
                tx_count_out=int(m.get("tx_count_out", 0)),
                tx_count_in=int(m.get("tx_count_in", 0)),
                total_value_eth=float(m.get("total_value_eth", 0)),
                last_seen_block=int(m.get("last_seen_block", 0)),
            )
            for m in result["members"]
        ]

        return ClusterDetail(
            cluster_id=cluster_data.get("cluster_id", id),
            behavioral_profile=cluster_data.get("behavioral_profile", "unknown"),
            member_count=int(cluster_data.get("member_count", 0)),
            total_mev_extracted_eth=float(
                cluster_data.get("total_mev_extracted_eth", 0)
            ),
            dominant_mev_type=cluster_data.get("dominant_mev_type"),
            members=members,
        )

    @strawberry.field(
        description="Aggregate MEV statistics for a range of blocks."
    )
    async def mev_summary(
        self, from_block: int, to_block: int, info: Info
    ) -> MEVSummary:
        timescale = info.context["timescale_client"]
        data = await timescale.get_mev_summary(from_block, to_block)

        by_type = [
            MEVByType(
                mev_type=r["mev_type"] or "unknown",
                count=int(r["count"]),
                extracted_eth=float(r["total_value_eth"] or 0),
            )
            for r in data["by_type"]
        ]

        top_actors = [
            TopActor(
                address=r["address"],
                extracted_eth=float(r["extracted_eth"] or 0),
            )
            for r in data["top_actors"]
        ]

        return MEVSummary(
            from_block=from_block,
            to_block=to_block,
            total_extracted_eth=float(data.get("total_extracted_eth", 0)),
            by_type=by_type,
            top_actors=top_actors,
        )

    @strawberry.field(
        description="List wallets that have interacted with a contract address."
    )
    async def who_touched(
        self,
        contract: str,
        limit: Optional[int] = 50,
        info: Info = None,
    ) -> List[ContractInteractor]:
        neo4j = info.context["neo4j_client"]
        rows = await neo4j.who_touched_contract(contract, limit=limit or 50)
        return [
            ContractInteractor(
                address=r["address"],
                interaction_count=int(r["interaction_count"]),
                last_block=int(r["last_block"] or 0),
                total_value_eth=float(r["total_value"] or 0),
                cluster_id=r.get("cluster_id"),
                cluster_profile=r.get("cluster_profile"),
            )
            for r in rows
        ]


schema = strawberry.Schema(query=Query)
