"""
NEPHILIM
========
Real-time on-chain entity intelligence and MEV attribution engine.

Entry point when run as ``python -m nephilim``.
Imports are lazy to avoid dragging in heavy dependencies during test collection.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Onur Kavi"


def main() -> None:
    """Sync wrapper called by ``python -m nephilim``."""
    import asyncio
    asyncio.run(_async_main())


async def _async_main() -> None:
    import sys
    from loguru import logger
    from nephilim.config import get_settings
    from nephilim.stream.block_subscriber import BlockSubscriber
    from nephilim.classifier.mev_classifier import MEVClassifier
    from nephilim.classifier.sandwich_detector import SandwichDetector
    from nephilim.classifier.jit_detector import JITDetector
    from nephilim.clustering.graph_builder import GraphBuilder
    from nephilim.clustering.entity_resolver import EntityResolver
    from nephilim.storage.neo4j_client import Neo4jClient
    from nephilim.storage.timescale_client import TimescaleClient
    from nephilim.alerts.telegram_alerter import TelegramAlerter

    settings = get_settings()

    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    logger.info("NEPHILIM v{} starting — chain_id={}", __version__, settings.chain_id)

    neo4j_client = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    timescale_client = TimescaleClient(dsn=settings.timescale_dsn)
    mev_classifier = MEVClassifier(model_path=settings.mev_classifier_model_path)
    sandwich_detector = SandwichDetector()
    jit_detector = JITDetector()
    graph_builder = GraphBuilder()
    entity_resolver = EntityResolver(graph_builder=graph_builder)
    alerter = TelegramAlerter(
        token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
    )

    await timescale_client.ensure_schema()
    await neo4j_client.ensure_constraints()

    subscriber = BlockSubscriber(
        ws_url=settings.alchemy_ws_url,
        mev_classifier=mev_classifier,
        sandwich_detector=sandwich_detector,
        jit_detector=jit_detector,
        graph_builder=graph_builder,
        entity_resolver=entity_resolver,
        neo4j_client=neo4j_client,
        timescale_client=timescale_client,
        alerter=alerter,
    )

    logger.info("Starting live block subscription…")
    await subscriber.run()


if __name__ == "__main__":
    main()
