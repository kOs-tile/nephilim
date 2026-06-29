"""
nephilim.storage.timescale_client
------------------------------------
TimescaleDB persistence for transaction time-series and MEV metrics.

Tables
------
``transactions``
    One row per decoded transaction. The ``block_timestamp`` column is the
    hypertable time dimension, enabling automatic chunk management and
    continuous aggregate queries.

``mev_block_summary``
    One row per block, aggregating MEV extraction statistics.

``mev_actor_hourly``
    Materialised continuous aggregate: MEV extracted per actor per hour.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import asyncpg
from loguru import logger

from nephilim.stream.transaction_decoder import TxRecord

_CREATE_TRANSACTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS transactions (
    block_timestamp     TIMESTAMPTZ NOT NULL,
    block_number        BIGINT      NOT NULL,
    tx_hash             TEXT        NOT NULL,
    position_in_block   INT         NOT NULL,
    from_address        TEXT        NOT NULL,
    to_address          TEXT,
    value_eth           DOUBLE PRECISION DEFAULT 0,
    value_usd           DOUBLE PRECISION DEFAULT 0,
    gas_price_gwei      DOUBLE PRECISION DEFAULT 0,
    gas_used            BIGINT DEFAULT 0,
    gas_price_percentile DOUBLE PRECISION DEFAULT 0,
    gas_premium_multiplier DOUBLE PRECISION DEFAULT 1,
    method_selector     TEXT,
    is_swap             BOOLEAN DEFAULT FALSE,
    is_lp_add           BOOLEAN DEFAULT FALSE,
    is_lp_remove        BOOLEAN DEFAULT FALSE,
    involves_flashloan  BOOLEAN DEFAULT FALSE,
    mev_type            TEXT,
    mev_confidence      DOUBLE PRECISION DEFAULT 0,
    PRIMARY KEY (block_timestamp, tx_hash)
);
"""

_CREATE_HYPERTABLE = """
SELECT create_hypertable(
    'transactions', 'block_timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 hour'
);
"""

_CREATE_MEV_BLOCK_SUMMARY_TABLE = """
CREATE TABLE IF NOT EXISTS mev_block_summary (
    block_timestamp     TIMESTAMPTZ NOT NULL,
    block_number        BIGINT      NOT NULL UNIQUE,
    tx_count            INT DEFAULT 0,
    sandwich_count      INT DEFAULT 0,
    jit_count           INT DEFAULT 0,
    total_mev_eth       DOUBLE PRECISION DEFAULT 0,
    PRIMARY KEY (block_timestamp, block_number)
);
"""

_CREATE_BLOCK_HYPERTABLE = """
SELECT create_hypertable(
    'mev_block_summary', 'block_timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);
"""

_INSERT_TRANSACTION = """
INSERT INTO transactions (
    block_timestamp, block_number, tx_hash, position_in_block,
    from_address, to_address, value_eth, value_usd,
    gas_price_gwei, gas_used, gas_price_percentile, gas_premium_multiplier,
    method_selector, is_swap, is_lp_add, is_lp_remove,
    involves_flashloan, mev_type, mev_confidence
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
    $11, $12, $13, $14, $15, $16, $17, $18, $19
)
ON CONFLICT (block_timestamp, tx_hash) DO NOTHING;
"""

_INSERT_BLOCK_SUMMARY = """
INSERT INTO mev_block_summary (
    block_timestamp, block_number, sandwich_count, jit_count, total_mev_eth
) VALUES ($1, $2, $3, $4, $5)
ON CONFLICT (block_number) DO UPDATE SET
    sandwich_count = EXCLUDED.sandwich_count,
    jit_count = EXCLUDED.jit_count,
    total_mev_eth = EXCLUDED.total_mev_eth;
"""


class TimescaleClient:
    """
    Async asyncpg client for TimescaleDB.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: Optional[asyncpg.Pool] = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                dsn=self._dsn,
                min_size=2,
                max_size=10,
                command_timeout=30,
            )
        return self._pool

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def ensure_schema(self) -> None:
        """Create tables and hypertables if they don't exist."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(_CREATE_TRANSACTIONS_TABLE)
            try:
                await conn.execute(_CREATE_HYPERTABLE)
            except asyncpg.exceptions.DuplicateTableError:
                pass
            except Exception as exc:  # noqa: BLE001
                logger.debug("Hypertable note: {}", exc)

            await conn.execute(_CREATE_MEV_BLOCK_SUMMARY_TABLE)
            try:
                await conn.execute(_CREATE_BLOCK_HYPERTABLE)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Block hypertable note: {}", exc)

        logger.info("TimescaleDB schema verified.")

    async def insert_transactions(self, txs: List[TxRecord]) -> None:
        """Batch insert classified transactions."""
        if not txs:
            return
        pool = await self._get_pool()
        now = datetime.now(timezone.utc)
        rows = [
            (
                now,
                tx.block_number,
                tx.hash,
                tx.position_in_block,
                tx.from_address,
                tx.to_address,
                tx.value_eth,
                tx.value_usd,
                tx.gas_price_wei / 1e9,
                tx.gas_used,
                tx.gas_price_percentile,
                tx.gas_premium_multiplier,
                tx.method_selector,
                tx.is_swap,
                tx.is_lp_add,
                tx.is_lp_remove,
                tx.involves_flashloan,
                tx.mev_type,
                tx.mev_confidence,
            )
            for tx in txs
        ]
        async with pool.acquire() as conn:
            await conn.executemany(_INSERT_TRANSACTION, rows)
        logger.debug("TimescaleDB: inserted {} transactions", len(rows))

    async def insert_mev_block_summary(
        self,
        block_number: int,
        sandwich_count: int,
        jit_count: int,
        total_mev_eth: float,
    ) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                _INSERT_BLOCK_SUMMARY,
                datetime.now(timezone.utc),
                block_number,
                sandwich_count,
                jit_count,
                total_mev_eth,
            )

    async def get_mev_summary(
        self, from_block: int, to_block: int
    ) -> Dict[str, Any]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    mev_type,
                    COUNT(*) AS count,
                    SUM(value_eth) AS total_value_eth
                FROM transactions
                WHERE block_number BETWEEN $1 AND $2
                  AND mev_type IS NOT NULL
                GROUP BY mev_type
                ORDER BY total_value_eth DESC
                """,
                from_block,
                to_block,
            )
            total = await conn.fetchval(
                """
                SELECT COALESCE(SUM(total_mev_eth), 0)
                FROM mev_block_summary
                WHERE block_number BETWEEN $1 AND $2
                """,
                from_block,
                to_block,
            )
            top_actors = await conn.fetch(
                """
                SELECT from_address AS address, SUM(value_eth) AS extracted_eth
                FROM transactions
                WHERE block_number BETWEEN $1 AND $2
                  AND mev_type NOT IN ('organic_swap', 'bridge')
                GROUP BY from_address
                ORDER BY extracted_eth DESC
                LIMIT 10
                """,
                from_block,
                to_block,
            )
        return {
            "total_extracted_eth": float(total or 0),
            "by_type": [dict(r) for r in rows],
            "top_actors": [dict(r) for r in top_actors],
        }

    async def get_wallet_mev_activity(
        self, address: str
    ) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    mev_type,
                    COUNT(*) AS count,
                    SUM(value_eth) AS total_value_eth,
                    MAX(block_number) AS last_block
                FROM transactions
                WHERE from_address = $1
                  AND mev_type IS NOT NULL
                GROUP BY mev_type
                ORDER BY count DESC
                """,
                address.lower(),
            )
        return [dict(r) for r in rows]
