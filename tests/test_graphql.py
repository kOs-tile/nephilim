"""
tests/test_graphql.py
-----------------------
Unit tests for GraphQL query resolution.

Uses Strawberry's ``schema.execute`` in synchronous mode with mock storage
clients — no live database connections required.

Tests cover:
1. ``wallet`` query returns correct fields when wallet exists.
2. ``wallet`` query returns None when wallet not found.
3. ``cluster`` query returns cluster detail with members.
4. ``mevSummary`` query returns correct aggregates.
5. ``whoTouched`` query returns correct interactors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
import strawberry
from strawberry.test import client as strawberry_client

from nephilim.api.graphql_schema import schema


# ── Mock storage clients ──────────────────────────────────────────────────────

class MockNeo4jClient:
    """In-memory mock of Neo4jClient."""

    def __init__(self) -> None:
        self._wallets: Dict[str, Dict[str, Any]] = {
            "0xaaaa": {
                "wallet": {
                    "address": "0xaaaa",
                    "label": "MEV-Bot-Alpha",
                    "behavioral_profile": "mev_bot",
                    "tx_count_out": 500,
                    "tx_count_in": 100,
                    "total_value_eth": 42.0,
                    "last_seen_block": 19_500_000,
                },
                "cluster": {
                    "cluster_id": "cluster-1",
                    "behavioral_profile": "mev_bot",
                    "member_count": 3,
                    "total_mev_extracted_eth": 120.0,
                    "dominant_mev_type": "sandwich_attack",
                },
            }
        }
        self._clusters: Dict[str, Dict[str, Any]] = {
            "cluster-1": {
                "cluster": {
                    "cluster_id": "cluster-1",
                    "behavioral_profile": "mev_bot",
                    "member_count": 2,
                    "total_mev_extracted_eth": 120.0,
                    "dominant_mev_type": "sandwich_attack",
                },
                "members": [
                    {
                        "address": "0xaaaa",
                        "tx_count_out": 500,
                        "tx_count_in": 100,
                        "total_value_eth": 42.0,
                        "last_seen_block": 19_500_000,
                    },
                    {
                        "address": "0xbbbb",
                        "tx_count_out": 300,
                        "tx_count_in": 80,
                        "total_value_eth": 28.0,
                        "last_seen_block": 19_499_900,
                    },
                ],
            }
        }

    async def get_wallet(self, address: str) -> Optional[Dict[str, Any]]:
        return self._wallets.get(address.lower())

    async def get_cluster(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        return self._clusters.get(cluster_id)

    async def who_touched_contract(
        self, contract_address: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        return [
            {
                "address": "0xaaaa",
                "interaction_count": 250,
                "last_block": 19_500_000,
                "total_value": 15.5,
                "cluster_id": "cluster-1",
                "cluster_profile": "mev_bot",
            },
            {
                "address": "0xcccc",
                "interaction_count": 5,
                "last_block": 19_499_000,
                "total_value": 0.5,
                "cluster_id": None,
                "cluster_profile": None,
            },
        ]


class MockTimescaleClient:
    """In-memory mock of TimescaleClient."""

    async def get_wallet_mev_activity(
        self, address: str
    ) -> List[Dict[str, Any]]:
        return [
            {"mev_type": "sandwich_attack", "count": 120, "total_value_eth": 8.5},
            {"mev_type": "cex_dex_arb", "count": 55, "total_value_eth": 3.2},
        ]

    async def get_mev_summary(
        self, from_block: int, to_block: int
    ) -> Dict[str, Any]:
        return {
            "total_extracted_eth": 42.5,
            "by_type": [
                {
                    "mev_type": "sandwich_attack",
                    "count": 18,
                    "total_value_eth": 22.1,
                },
                {
                    "mev_type": "cex_dex_arb",
                    "count": 9,
                    "total_value_eth": 10.4,
                },
            ],
            "top_actors": [
                {"address": "0xaaaa", "extracted_eth": 18.3},
                {"address": "0xbbbb", "extracted_eth": 12.1},
            ],
        }


# ── Test context factory ──────────────────────────────────────────────────────

_neo4j = MockNeo4jClient()
_timescale = MockTimescaleClient()


async def _context() -> Dict[str, Any]:
    return {"neo4j_client": _neo4j, "timescale_client": _timescale}


# ── Helper ────────────────────────────────────────────────────────────────────

async def _execute(query: str, variables: Optional[Dict[str, Any]] = None) -> Any:
    result = await schema.execute(
        query,
        context_value=await _context(),
        variable_values=variables or {},
    )
    return result


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestWalletQuery:
    async def test_wallet_found(self) -> None:
        result = await _execute(
            """
            query {
              wallet(address: "0xaaaa") {
                address
                label
                behavioralProfile
                txCountOut
                cluster {
                  clusterId
                  behavioralProfile
                }
                mevActivity {
                  mevType
                  count
                }
              }
            }
            """
        )
        assert result.errors is None
        data = result.data["wallet"]
        assert data["address"] == "0xaaaa"
        assert data["label"] == "MEV-Bot-Alpha"
        assert data["behavioralProfile"] == "mev_bot"
        assert data["txCountOut"] == 500
        assert data["cluster"]["clusterId"] == "cluster-1"
        assert len(data["mevActivity"]) == 2
        types = {a["mevType"] for a in data["mevActivity"]}
        assert "sandwich_attack" in types

    async def test_wallet_not_found_returns_null(self) -> None:
        result = await _execute(
            """
            query {
              wallet(address: "0xdeadbeef") {
                address
              }
            }
            """
        )
        assert result.errors is None
        assert result.data["wallet"] is None


@pytest.mark.asyncio
class TestClusterQuery:
    async def test_cluster_found(self) -> None:
        result = await _execute(
            """
            query {
              cluster(id: "cluster-1") {
                clusterId
                behavioralProfile
                memberCount
                dominantMevType
                members {
                  address
                  txCountOut
                }
              }
            }
            """
        )
        assert result.errors is None
        data = result.data["cluster"]
        assert data["clusterId"] == "cluster-1"
        assert data["behavioralProfile"] == "mev_bot"
        assert data["dominantMevType"] == "sandwich_attack"
        assert len(data["members"]) == 2

    async def test_cluster_not_found_returns_null(self) -> None:
        result = await _execute(
            """
            query {
              cluster(id: "nonexistent-cluster") {
                clusterId
              }
            }
            """
        )
        assert result.errors is None
        assert result.data["cluster"] is None


@pytest.mark.asyncio
class TestMEVSummaryQuery:
    async def test_mev_summary_returns_aggregates(self) -> None:
        result = await _execute(
            """
            query {
              mevSummary(fromBlock: 19000000, toBlock: 19500000) {
                fromBlock
                toBlock
                totalExtractedEth
                byType {
                  mevType
                  count
                  extractedEth
                }
                topActors {
                  address
                  extractedEth
                }
              }
            }
            """
        )
        assert result.errors is None
        data = result.data["mevSummary"]
        assert data["fromBlock"] == 19_000_000
        assert data["toBlock"] == 19_500_000
        assert data["totalExtractedEth"] == pytest.approx(42.5)
        assert len(data["byType"]) == 2
        assert len(data["topActors"]) == 2
        top = data["topActors"][0]
        assert top["address"] == "0xaaaa"
        assert top["extractedEth"] == pytest.approx(18.3)


@pytest.mark.asyncio
class TestWhoTouchedQuery:
    async def test_who_touched_returns_interactors(self) -> None:
        result = await _execute(
            """
            query {
              whoTouched(contract: "0xe592427a0aece92de3edee1f18e0157c05861564") {
                address
                interactionCount
                lastBlock
                totalValueEth
                clusterId
                clusterProfile
              }
            }
            """
        )
        assert result.errors is None
        data = result.data["whoTouched"]
        assert len(data) == 2
        top = data[0]
        assert top["address"] == "0xaaaa"
        assert top["interactionCount"] == 250
        assert top["clusterId"] == "cluster-1"
        assert top["clusterProfile"] == "mev_bot"

        second = data[1]
        assert second["clusterId"] is None
