"""
nephilim.stream.block_subscriber
---------------------------------
Async WebSocket subscriber to Alchemy/QuickNode.

Receives new block headers via eth_subscribe("newHeads"), fetches full
transaction data with receipts, decodes each transaction, then drives the
full MEV classification + graph-update pipeline.

Each block is also published to the Kafka ``raw_blocks`` and ``mev_events``
topics for downstream consumers.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import websockets
from kafka import KafkaProducer
from loguru import logger
from tenacity import (
    AsyncRetrying,
    RetryError,
    stop_after_attempt,
    wait_exponential,
)
from web3 import AsyncWeb3
from web3.providers.persistent import WebSocketProvider as WebsocketProviderV2

from nephilim.config import get_settings
from nephilim.stream.transaction_decoder import TransactionDecoder, TxRecord

if TYPE_CHECKING:
    from nephilim.alerts.telegram_alerter import TelegramAlerter
    from nephilim.classifier.jit_detector import JITDetector
    from nephilim.classifier.mev_classifier import MEVClassifier
    from nephilim.classifier.sandwich_detector import SandwichDetector
    from nephilim.clustering.entity_resolver import EntityResolver
    from nephilim.clustering.graph_builder import GraphBuilder
    from nephilim.storage.neo4j_client import Neo4jClient
    from nephilim.storage.timescale_client import TimescaleClient


class BlockSubscriber:
    """
    Drives the end-to-end NEPHILIM pipeline for every new block:

    1. Subscribe to ``newHeads`` via WebSocket.
    2. Fetch full block + receipts.
    3. Decode each transaction.
    4. Run sandwich / JIT pattern detectors.
    5. Classify each transaction with the XGBoost model.
    6. Update the entity graph.
    7. Persist to TimescaleDB and Neo4j.
    8. Fire Telegram alerts for significant MEV events.
    """

    _RECONNECT_DELAY = 5.0  # seconds before reconnect on WS drop

    def __init__(
        self,
        ws_url: str,
        mev_classifier: "MEVClassifier",
        sandwich_detector: "SandwichDetector",
        jit_detector: "JITDetector",
        graph_builder: "GraphBuilder",
        entity_resolver: "EntityResolver",
        neo4j_client: "Neo4jClient",
        timescale_client: "TimescaleClient",
        alerter: "TelegramAlerter",
    ) -> None:
        self._ws_url = ws_url
        self._mev_classifier = mev_classifier
        self._sandwich_detector = sandwich_detector
        self._jit_detector = jit_detector
        self._graph_builder = graph_builder
        self._entity_resolver = entity_resolver
        self._neo4j_client = neo4j_client
        self._timescale_client = timescale_client
        self._alerter = alerter
        self._settings = get_settings()

        self._decoder: Optional[TransactionDecoder] = None
        self._producer: Optional[KafkaProducer] = None
        self._blocks_processed = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Run forever, reconnecting on WebSocket drops."""
        self._producer = self._build_kafka_producer()
        logger.info("Kafka producer initialised at {}", self._settings.kafka_brokers)

        while True:
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(10),
                    wait=wait_exponential(multiplier=1, min=2, max=30),
                ):
                    with attempt:
                        await self._connect_and_stream()
            except RetryError:
                logger.error("Exhausted reconnect attempts — giving up.")
                return
            except asyncio.CancelledError:
                logger.info("BlockSubscriber cancelled.")
                return

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_kafka_producer(self) -> KafkaProducer:
        return KafkaProducer(
            bootstrap_servers=self._settings.kafka_broker_list,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=3,
        )

    async def _connect_and_stream(self) -> None:
        logger.info("Connecting to WebSocket: {}", self._ws_url)

        async with AsyncWeb3(
            WebsocketProviderV2(
                self._ws_url,
                websocket_kwargs={"ping_interval": 20, "ping_timeout": 10},
            )
        ) as w3:
            self._decoder = TransactionDecoder(w3=w3, redis_url=self._settings.redis_url)
            logger.info("WebSocket connected — awaiting new blocks…")

            subscription_id = await w3.eth.subscribe("newHeads")
            logger.debug("Subscribed with id={}", subscription_id)

            async for payload in w3.socket.process_subscriptions():
                block_header: Dict[str, Any] = payload["result"]
                block_number = int(block_header["number"], 16)
                logger.info("New block #{}", block_number)
                await self._process_block(w3, block_number)

    async def _process_block(
        self, w3: AsyncWeb3, block_number: int
    ) -> None:
        """Fetch, decode, classify, and store a single block."""
        try:
            block = await w3.eth.get_block(block_number, full_transactions=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch block #{}: {}", block_number, exc)
            return

        raw_txs = block.get("transactions", [])
        if not raw_txs:
            logger.debug("Block #{} has no transactions — skipping.", block_number)
            return

        logger.debug(
            "Block #{} — {} transactions to process", block_number, len(raw_txs)
        )

        # Publish raw block to Kafka
        self._publish_kafka(
            self._settings.kafka_topic_raw_blocks,
            {
                "block_number": block_number,
                "timestamp": int(block["timestamp"]),
                "tx_count": len(raw_txs),
                "base_fee": int(block.get("baseFeePerGas", 0)),
            },
        )

        # Decode transactions
        decoded_txs: List[TxRecord] = []
        gas_prices = [int(tx["gasPrice"]) for tx in raw_txs if "gasPrice" in tx]
        gas_prices.sort()

        for idx, raw_tx in enumerate(raw_txs):
            try:
                tx_record = await self._decoder.decode(  # type: ignore[union-attr]
                    raw_tx=raw_tx,
                    block_number=block_number,
                    position_in_block=idx,
                    block_base_fee=int(block.get("baseFeePerGas", 0)),
                    block_gas_prices=gas_prices,
                )
                decoded_txs.append(tx_record)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to decode tx {} in block #{}: {}",
                    raw_tx.get("hash", "?"),
                    block_number,
                    exc,
                )

        # Pattern detection (block-level)
        sandwich_attacks = self._sandwich_detector.detect(decoded_txs)
        jit_events = self._jit_detector.detect(decoded_txs)

        # Mark detected patterns on the tx records
        sandwich_hashes = {
            h for attack in sandwich_attacks for h in attack["tx_hashes"]
        }
        jit_hashes = {h for event in jit_events for h in event["tx_hashes"]}

        for tx in decoded_txs:
            if tx.hash in sandwich_hashes:
                tx.pattern_flags.append("sandwich")
            if tx.hash in jit_hashes:
                tx.pattern_flags.append("jit_liquidity")

        # ML classification
        classified_txs = [
            self._mev_classifier.predict(tx) for tx in decoded_txs
        ]

        # Build / update graph
        for tx in classified_txs:
            self._graph_builder.add_transaction(tx)

        # Resolve entities every N blocks
        if self._blocks_processed % 10 == 0:
            entity_updates = self._entity_resolver.resolve()
            await self._neo4j_client.upsert_entities(entity_updates)

        # Persist to TimescaleDB
        await self._timescale_client.insert_transactions(classified_txs)
        await self._timescale_client.insert_mev_block_summary(
            block_number=block_number,
            sandwich_count=len(sandwich_attacks),
            jit_count=len(jit_events),
            total_mev_eth=sum(
                a["extracted_value_eth"] for a in sandwich_attacks
            ) + sum(e.get("extracted_value_eth", 0.0) for e in jit_events),
        )

        # Fire alerts
        for attack in sandwich_attacks:
            await self._alerter.send_mev_alert(
                mev_type="sandwich_attack",
                block_number=block_number,
                extracted_eth=attack["extracted_value_eth"],
                actor_address=attack["attacker"],
                details=attack,
            )

        # Publish MEV events to Kafka
        for attack in sandwich_attacks:
            self._publish_kafka(self._settings.kafka_topic_mev_events, {
                "type": "sandwich_attack",
                "block": block_number,
                **attack,
            })

        self._blocks_processed += 1
        logger.success(
            "Block #{} processed — {} txs, {} sandwiches, {} JIT events",
            block_number,
            len(decoded_txs),
            len(sandwich_attacks),
            len(jit_events),
        )

    def _publish_kafka(self, topic: str, payload: Dict[str, Any]) -> None:
        if self._producer:
            try:
                self._producer.send(topic, value=payload)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Kafka publish failed on topic {}: {}", topic, exc)
