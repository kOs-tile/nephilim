"""
nephilim.config
---------------
Centralised configuration via Pydantic Settings.
All values are loaded from environment variables or a .env file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Chain / Node ──────────────────────────────────────────────────────────
    alchemy_ws_url: str = Field(
        default="wss://eth-mainnet.g.alchemy.com/v2/demo",
        description="Alchemy WebSocket endpoint for Ethereum mainnet",
    )
    quicknode_arb_ws_url: str | None = Field(
        default=None,
        description="Optional QuickNode WebSocket for Arbitrum One",
    )
    chain_id: int = Field(default=1, description="EVM chain ID (1=mainnet, 42161=Arb)")

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="nephilim_secret")

    # ── TimescaleDB ───────────────────────────────────────────────────────────
    timescale_dsn: str = Field(
        default="postgresql://nephilim:nephilim_secret@localhost:5432/nephilim"
    )

    # ── Kafka ─────────────────────────────────────────────────────────────────
    kafka_brokers: str = Field(default="localhost:9092")
    kafka_topic_raw_blocks: str = Field(default="raw_blocks")
    kafka_topic_decoded_txs: str = Field(default="decoded_txs")
    kafka_topic_mev_events: str = Field(default="mev_events")

    @property
    def kafka_broker_list(self) -> List[str]:
        return [b.strip() for b in self.kafka_brokers.split(",")]

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0")
    token_price_cache_ttl: int = Field(default=30, description="seconds")

    # ── Telegram ──────────────────────────────────────────────────────────────
    telegram_bot_token: str | None = Field(default=None)
    telegram_chat_id: str | None = Field(default=None)
    alert_min_mev_eth: float = Field(
        default=1.0, description="Minimum ETH value to trigger alert"
    )

    # ── Model Paths ───────────────────────────────────────────────────────────
    mev_classifier_model_path: Path = Field(default=Path("models/mev_classifier.pkl"))
    mev_classifier_label_encoder_path: Path = Field(
        default=Path("models/label_encoder.pkl")
    )

    # ── Application ───────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    backfill_blocks: int = Field(
        default=0, description="Blocks to backfill on startup (0 = live only)"
    )
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return upper


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
