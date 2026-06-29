"""
nephilim.stream.transaction_decoder
-------------------------------------
Decode raw Ethereum transactions into structured ``TxRecord`` objects.

Responsibilities:
- Extract from / to / value / input data from raw transaction dict.
- Identify DEX router interactions using known Uniswap V2/V3 method signatures.
- Look up token USD prices from Redis cache (populated by a price oracle worker).
- Compute ``gas_price_percentile`` relative to the current block's gas distribution.
- Flag flashloan, oracle-touching, and governance interactions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import redis.asyncio as aioredis
from loguru import logger
from web3 import AsyncWeb3

# ── Known DEX / Protocol selectors (first 4 bytes of keccak256(signature)) ───

# Uniswap V2 Router
_UNI_V2_SWAP_ETH_FOR_TOKENS = "0x7ff36ab5"
_UNI_V2_SWAP_TOKENS_FOR_ETH = "0x18cbafe5"
_UNI_V2_SWAP_TOKENS_FOR_TOKENS = "0x38ed1739"
_UNI_V2_SWAP_EXACT_ETH = "0xb6f9de95"

# Uniswap V3 Router / SwapRouter
_UNI_V3_EXACT_INPUT = "0x414bf389"   # exactInputSingle
_UNI_V3_EXACT_INPUT_MULTI = "0xc04b8d59"  # exactInput
_UNI_V3_EXACT_OUTPUT = "0xdb3e2198"  # exactOutputSingle

# Uniswap V3 NonfungiblePositionManager (JIT signals)
_UNI_V3_MINT = "0x88316456"          # mint(params)
_UNI_V3_COLLECT = "0xfc6f7865"       # collect(params)
_UNI_V3_DECREASE_LIQUIDITY = "0x0c49ccbe"

# Flash loan providers
_AAVE_FLASH_LOAN = "0xab9c4b5d"      # flashLoan (v2)
_AAVE_FLASH_LOAN_SIMPLE = "0x42b0b77c"
_BALANCER_FLASH_LOAN = "0x5c38449e"

# Oracle
_CHAINLINK_LATEST_ROUND = "0xfeaf968c"
_UNISWAP_OBSERVE = "0x883bdbfd"      # TWAP observation

# Governance
_COMPOUND_CAST_VOTE = "0x15373e3d"
_OPENZEPPELIN_CAST_VOTE = "0x56781388"
_BRAVO_PROPOSE = "0xda95691a"

_SWAP_SELECTORS = {
    _UNI_V2_SWAP_ETH_FOR_TOKENS,
    _UNI_V2_SWAP_TOKENS_FOR_ETH,
    _UNI_V2_SWAP_TOKENS_FOR_TOKENS,
    _UNI_V2_SWAP_EXACT_ETH,
    _UNI_V3_EXACT_INPUT,
    _UNI_V3_EXACT_INPUT_MULTI,
    _UNI_V3_EXACT_OUTPUT,
}

_LP_ADD_SELECTORS = {_UNI_V3_MINT}
_LP_REMOVE_SELECTORS = {_UNI_V3_DECREASE_LIQUIDITY, _UNI_V3_COLLECT}
_FLASHLOAN_SELECTORS = {_AAVE_FLASH_LOAN, _AAVE_FLASH_LOAN_SIMPLE, _BALANCER_FLASH_LOAN}
_ORACLE_SELECTORS = {_CHAINLINK_LATEST_ROUND, _UNISWAP_OBSERVE}
_GOVERNANCE_SELECTORS = {_COMPOUND_CAST_VOTE, _OPENZEPPELIN_CAST_VOTE, _BRAVO_PROPOSE}

# Well-known Uniswap V3 factory / router addresses (lowercase)
_UNISWAP_V3_ROUTERS = {
    "0xe592427a0aece92de3edee1f18e0157c05861564",
    "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",
    "0xef1c6e67703c7bd7107eed8303fbe6ec2554bf6b",
}
_UNISWAP_V2_ROUTERS = {
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",
    "0xf491e7b69e4244ad4002bc14e878a34207e38c29",
}


@dataclass
class TxRecord:
    """Fully decoded transaction with feature-engineered fields for the ML classifier."""

    # Core fields
    hash: str
    block_number: int
    position_in_block: int
    from_address: str
    to_address: Optional[str]
    value_wei: int
    value_eth: float
    value_usd: float
    gas_price_wei: int
    gas_used: int
    input_data: str
    method_selector: str

    # DEX interaction flags
    is_swap: bool = False
    is_lp_add: bool = False
    is_lp_remove: bool = False
    involves_uniswap_v3: bool = False
    involves_uniswap_v2: bool = False

    # MEV feature signals
    involves_flashloan: bool = False
    touches_price_oracle: bool = False
    is_governance: bool = False

    # Block-level contextual features (set by BlockSubscriber)
    gas_price_percentile: float = 0.0  # 0–100 relative to block
    same_block_counterpart: bool = False  # paired tx in same block
    block_base_fee_wei: int = 0

    # Computed features
    gas_premium_multiplier: float = 1.0  # gas_price / base_fee
    priority_fee_wei: int = 0

    # Pattern flags applied by block-level detectors
    pattern_flags: List[str] = field(default_factory=list)

    # Classification result (set by MEVClassifier.predict)
    mev_type: Optional[str] = None
    mev_confidence: float = 0.0
    extracted_value_eth: float = 0.0


class TransactionDecoder:
    """
    Decodes raw web3.py transaction dicts into ``TxRecord`` objects.

    Token USD prices are fetched from a Redis cache (expected key format:
    ``price:{token_address_lower}`` → JSON-encoded float).  When the price is
    absent the USD value falls back to 0.0 rather than blocking the pipeline.
    """

    _ETH_USD_KEY = "price:eth"

    def __init__(self, w3: AsyncWeb3, redis_url: str) -> None:
        self._w3 = w3
        self._redis: Optional[aioredis.Redis] = None
        self._redis_url = redis_url
        self._eth_price_cache: float = 0.0

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = await aioredis.from_url(
                self._redis_url, encoding="utf-8", decode_responses=True
            )
        return self._redis

    async def get_eth_price_usd(self) -> float:
        try:
            r = await self._get_redis()
            raw = await r.get(self._ETH_USD_KEY)
            if raw:
                self._eth_price_cache = float(raw)
        except Exception:  # noqa: BLE001
            pass
        return self._eth_price_cache if self._eth_price_cache else 3500.0  # fallback

    async def decode(
        self,
        raw_tx: Dict[str, Any],
        block_number: int,
        position_in_block: int,
        block_base_fee: int,
        block_gas_prices: List[int],
    ) -> TxRecord:
        """
        Convert a raw transaction dict (from web3 ``getBlock(full_transactions=True)``)
        into a richly-annotated ``TxRecord``.
        """
        tx_hash = raw_tx.get("hash", "")
        if isinstance(tx_hash, (bytes, bytearray)):
            tx_hash = tx_hash.hex()

        from_addr = str(raw_tx.get("from", "")).lower()
        to_addr = raw_tx.get("to")
        to_addr = str(to_addr).lower() if to_addr else None

        value_wei = int(raw_tx.get("value", 0))
        value_eth = value_wei / 1e18
        eth_price = await self.get_eth_price_usd()
        value_usd = value_eth * eth_price

        gas_price_wei = int(raw_tx.get("gasPrice", 0))
        gas_used = int(raw_tx.get("gas", 0))

        input_data: str = raw_tx.get("input", "0x") or "0x"
        method_selector = input_data[:10].lower() if len(input_data) >= 10 else "0x"

        # Gas percentile relative to block
        gas_price_percentile = self._compute_gas_percentile(
            gas_price_wei, block_gas_prices
        )

        # Gas premium
        gas_premium_multiplier = (
            gas_price_wei / block_base_fee if block_base_fee else 1.0
        )
        priority_fee_wei = max(0, gas_price_wei - block_base_fee)

        # Interaction flags
        is_swap = method_selector in _SWAP_SELECTORS
        is_lp_add = method_selector in _LP_ADD_SELECTORS
        is_lp_remove = method_selector in _LP_REMOVE_SELECTORS
        involves_flashloan = method_selector in _FLASHLOAN_SELECTORS
        touches_price_oracle = method_selector in _ORACLE_SELECTORS
        is_governance = method_selector in _GOVERNANCE_SELECTORS
        involves_uniswap_v3 = to_addr in _UNISWAP_V3_ROUTERS
        involves_uniswap_v2 = to_addr in _UNISWAP_V2_ROUTERS

        return TxRecord(
            hash=tx_hash,
            block_number=block_number,
            position_in_block=position_in_block,
            from_address=from_addr,
            to_address=to_addr,
            value_wei=value_wei,
            value_eth=value_eth,
            value_usd=value_usd,
            gas_price_wei=gas_price_wei,
            gas_used=gas_used,
            input_data=input_data,
            method_selector=method_selector,
            is_swap=is_swap,
            is_lp_add=is_lp_add,
            is_lp_remove=is_lp_remove,
            involves_uniswap_v3=involves_uniswap_v3,
            involves_uniswap_v2=involves_uniswap_v2,
            involves_flashloan=involves_flashloan,
            touches_price_oracle=touches_price_oracle,
            is_governance=is_governance,
            gas_price_percentile=gas_price_percentile,
            block_base_fee_wei=block_base_fee,
            gas_premium_multiplier=gas_premium_multiplier,
            priority_fee_wei=priority_fee_wei,
        )

    @staticmethod
    def _compute_gas_percentile(gas_price: int, block_gas_prices: List[int]) -> float:
        """Return the percentile rank (0-100) of gas_price within block_gas_prices."""
        if not block_gas_prices:
            return 50.0
        below = sum(1 for g in block_gas_prices if g <= gas_price)
        return round(100.0 * below / len(block_gas_prices), 2)
