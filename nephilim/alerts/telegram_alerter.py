"""
nephilim.alerts.telegram_alerter
-----------------------------------
Real-time Telegram alert dispatcher for significant MEV events.

Alert triggers
--------------
1. **Sandwich attack** with extracted value ≥ ``ALERT_MIN_MEV_ETH``
2. **New cluster activation** on a protocol the cluster hasn't touched before
3. **Flashloan-based attack** over threshold
4. **Oracle manipulation** detected

Message format
--------------
Each alert is a rich formatted Telegram message (HTML parse mode) that
includes:
- MEV type badge
- Block number and tx hashes (Etherscan links)
- Extracted value in ETH
- Attacker address (cluster label if resolved)
- Cluster history summary

Rate limiting: at most 1 alert per attacker per 5 minutes to avoid flooding.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

from loguru import logger

try:
    from telegram import Bot
    from telegram.constants import ParseMode
    _TELEGRAM_AVAILABLE = True
except ImportError:
    _TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed — Telegram alerts disabled.")


# MEV type → human-readable label + emoji
_MEV_LABELS: Dict[str, str] = {
    "sandwich_attack": "🥪 Sandwich Attack",
    "jit_liquidity": "⚡ JIT Liquidity",
    "cex_dex_arb": "📈 CEX-DEX Arb",
    "pure_arb": "🔄 Pure Arb",
    "liquidation": "💀 Liquidation",
    "nft_sweep": "🖼 NFT Sweep",
    "wash_trade": "🔃 Wash Trade",
    "flashloan_attack": "🔦 Flashloan Attack",
    "oracle_manipulation": "🔮 Oracle Manipulation",
    "governance_attack": "🗳 Governance Attack",
}

_ETHERSCAN_TX = "https://etherscan.io/tx/{}"
_ETHERSCAN_ADDR = "https://etherscan.io/address/{}"

# Rate limit: seconds between alerts for the same attacker address
_RATE_LIMIT_SECONDS = 300


class TelegramAlerter:
    """
    Sends MEV alerts to a Telegram chat via the Bot API.

    If ``token`` or ``chat_id`` is None, the alerter operates in a silent
    no-op mode — no exceptions are raised, making it safe to run in
    environments without Telegram credentials.
    """

    def __init__(
        self,
        token: Optional[str],
        chat_id: Optional[str],
        min_mev_eth: float = 1.0,
    ) -> None:
        self._token = token
        self._chat_id = chat_id
        self._min_mev_eth = min_mev_eth
        self._bot: Optional["Bot"] = None
        self._last_alert: Dict[str, float] = {}  # address → timestamp

        if token and _TELEGRAM_AVAILABLE:
            self._bot = Bot(token=token)
            logger.info("TelegramAlerter initialised (chat_id={})", chat_id)
        else:
            logger.info(
                "TelegramAlerter in no-op mode "
                "(token={}, telegram_available={})",
                bool(token),
                _TELEGRAM_AVAILABLE,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    async def send_mev_alert(
        self,
        mev_type: str,
        block_number: int,
        extracted_eth: float,
        actor_address: str,
        details: Dict[str, Any],
        cluster_profile: Optional[str] = None,
        cluster_id: Optional[str] = None,
        cluster_history_summary: Optional[str] = None,
    ) -> None:
        """
        Fire an alert if:
        - The bot is configured.
        - Extracted value meets the threshold.
        - The actor hasn't been alerted recently (rate limit).
        """
        if not self._bot or not self._chat_id:
            return

        if extracted_eth < self._min_mev_eth:
            return

        if self._is_rate_limited(actor_address):
            logger.debug(
                "Alert suppressed for {} (rate limited)", actor_address[:10]
            )
            return

        message = self._format_alert(
            mev_type=mev_type,
            block_number=block_number,
            extracted_eth=extracted_eth,
            actor_address=actor_address,
            details=details,
            cluster_profile=cluster_profile,
            cluster_id=cluster_id,
            cluster_history_summary=cluster_history_summary,
        )

        try:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=message,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
            self._mark_alerted(actor_address)
            logger.info(
                "Telegram alert sent: {} — {:.4f} ETH extracted by {}",
                mev_type,
                extracted_eth,
                actor_address[:10],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Telegram send failed: {}", exc)

    async def send_cluster_alert(
        self,
        cluster_id: str,
        behavioral_profile: str,
        new_protocol: str,
        total_extracted_eth: float,
        member_count: int,
    ) -> None:
        """Alert when a known MEV cluster activates on a new protocol."""
        if not self._bot or not self._chat_id:
            return

        message = (
            f"<b>🕵️ Known Cluster Active on New Protocol</b>\n\n"
            f"<b>Cluster:</b> <code>{cluster_id}</code>\n"
            f"<b>Profile:</b> {behavioral_profile}\n"
            f"<b>New Protocol:</b> {new_protocol}\n"
            f"<b>Members:</b> {member_count}\n"
            f"<b>Total Extracted (historical):</b> "
            f"{total_extracted_eth:.4f} ETH\n\n"
            f"<i>NEPHILIM detected first interaction with this protocol.</i>"
        )

        try:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=message,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Telegram cluster alert failed: {}", exc)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _format_alert(
        self,
        mev_type: str,
        block_number: int,
        extracted_eth: float,
        actor_address: str,
        details: Dict[str, Any],
        cluster_profile: Optional[str],
        cluster_id: Optional[str],
        cluster_history_summary: Optional[str],
    ) -> str:
        label = _MEV_LABELS.get(mev_type, f"⚠️ {mev_type}")
        short_actor = f"{actor_address[:6]}...{actor_address[-4:]}"
        etherscan_addr = _ETHERSCAN_ADDR.format(actor_address)

        tx_hashes = details.get("tx_hashes", [])
        tx_links = "\n".join(
            f"  • <a href='{_ETHERSCAN_TX.format(h)}'>{h[:10]}...</a>"
            for h in tx_hashes[:3]
        )

        victim_info = ""
        if "victim" in details:
            victim = details["victim"]
            victim_short = f"{victim[:6]}...{victim[-4:]}"
            victim_info = (
                f"\n<b>Victim:</b> "
                f"<a href='{_ETHERSCAN_ADDR.format(victim)}'>{victim_short}</a>"
            )

        cluster_info = ""
        if cluster_id:
            cluster_info = (
                f"\n<b>Cluster:</b> <code>{cluster_id}</code> "
                f"({cluster_profile or 'unknown'})"
            )
        if cluster_history_summary:
            cluster_info += f"\n<i>{cluster_history_summary}</i>"

        return (
            f"<b>{label}</b>\n\n"
            f"<b>Block:</b> #{block_number:,}\n"
            f"<b>Extracted:</b> {extracted_eth:.6f} ETH\n"
            f"<b>Attacker:</b> "
            f"<a href='{etherscan_addr}'>{short_actor}</a>"
            f"{victim_info}"
            f"{cluster_info}\n\n"
            f"<b>Transactions:</b>\n{tx_links}\n\n"
            f"<i>Detected by NEPHILIM</i>"
        )

    def _is_rate_limited(self, address: str) -> bool:
        last = self._last_alert.get(address, 0.0)
        return (time.monotonic() - last) < _RATE_LIMIT_SECONDS

    def _mark_alerted(self, address: str) -> None:
        self._last_alert[address] = time.monotonic()
