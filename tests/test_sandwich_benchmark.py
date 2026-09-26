from nephilim.benchmark.sandwich import (
    SandwichBenchmarkCase,
    evaluate_sandwich_detector,
)
from nephilim.stream.transaction_decoder import TxRecord


BOT = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
VICTIM = "0x1111111111111111111111111111111111111111"
OTHER = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
TOKEN_A = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1"
TOKEN_B = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb2"
PAIR = f"uniswap_v3:{TOKEN_A}:{TOKEN_B}:3000"


def make_tx(
    tx_hash: str,
    position: int,
    sender: str,
    gas_gwei: float,
    *,
    pair: str | None = PAIR,
    token_in: str | None = TOKEN_A,
    token_out: str | None = TOKEN_B,
) -> TxRecord:
    gas_wei = int(gas_gwei * 1e9)
    return TxRecord(
        hash=tx_hash,
        block_number=1,
        position_in_block=position,
        from_address=sender,
        to_address="0xe592427a0aece92de3edee1f18e0157c05861564",
        value_wei=int(5e18),
        value_eth=5.0,
        value_usd=0.0,
        gas_price_wei=gas_wei,
        gas_used=150_000,
        input_data="0x414bf389",
        method_selector="0x414bf389",
        is_swap=True,
        dex_pair_key=pair,
        dex_token_in=token_in,
        dex_token_out=token_out,
    )


def canonical_attack() -> list[TxRecord]:
    return [
        make_tx("front", 0, BOT, 45),
        make_tx("victim", 1, VICTIM, 20),
        make_tx(
            "back",
            2,
            BOT,
            45,
            token_in=TOKEN_B,
            token_out=TOKEN_A,
        ),
    ]


def test_synthetic_adversarial_benchmark_has_no_regression_false_positives():
    other_pair = f"uniswap_v3:{TOKEN_A}:0xccccccccccccccccccccccccccccccccccccccc3:3000"
    cases = [
        SandwichBenchmarkCase(
            name="canonical directional sandwich",
            txs=canonical_attack(),
            expected_detected=True,
        ),
        SandwichBenchmarkCase(
            name="same-direction attacker round trip is not a sandwich",
            txs=[
                make_tx("front-same", 0, BOT, 45),
                make_tx("victim-same", 1, VICTIM, 20),
                make_tx("back-same", 2, BOT, 45),
            ],
            expected_detected=False,
        ),
        SandwichBenchmarkCase(
            name="missing direction evidence fails closed",
            txs=[
                make_tx("front-missing", 0, BOT, 45),
                make_tx(
                    "victim-missing",
                    1,
                    VICTIM,
                    20,
                    token_in=None,
                    token_out=None,
                ),
                make_tx(
                    "back-missing",
                    2,
                    BOT,
                    45,
                    token_in=TOKEN_B,
                    token_out=TOKEN_A,
                ),
            ],
            expected_detected=False,
        ),
        SandwichBenchmarkCase(
            name="same gas ordering is rejected",
            txs=[
                make_tx("front-gas", 0, BOT, 20),
                make_tx("victim-gas", 1, VICTIM, 20),
                make_tx(
                    "back-gas",
                    2,
                    BOT,
                    20,
                    token_in=TOKEN_B,
                    token_out=TOKEN_A,
                ),
            ],
            expected_detected=False,
        ),
        SandwichBenchmarkCase(
            name="different pair victim is rejected",
            txs=[
                make_tx("front-pair", 0, BOT, 45),
                make_tx(
                    "victim-pair",
                    1,
                    VICTIM,
                    20,
                    pair=other_pair,
                    token_out="0xccccccccccccccccccccccccccccccccccccccc3",
                ),
                make_tx(
                    "back-pair",
                    2,
                    BOT,
                    45,
                    token_in=TOKEN_B,
                    token_out=TOKEN_A,
                ),
            ],
            expected_detected=False,
        ),
        SandwichBenchmarkCase(
            name="different backrun sender is rejected",
            txs=[
                make_tx("front-sender", 0, BOT, 45),
                make_tx("victim-sender", 1, VICTIM, 20),
                make_tx(
                    "back-sender",
                    2,
                    OTHER,
                    45,
                    token_in=TOKEN_B,
                    token_out=TOKEN_A,
                ),
            ],
            expected_detected=False,
        ),
    ]

    result = evaluate_sandwich_detector(cases)

    assert result.total_cases == 6
    assert result.true_positives == 1
    assert result.false_positives == 0
    assert result.true_negatives == 5
    assert result.false_negatives == 0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.false_positive_rate == 0.0
    assert result.accuracy == 1.0


def test_benchmark_result_serializes_metrics():
    result = evaluate_sandwich_detector(
        [
            SandwichBenchmarkCase(
                name="positive",
                txs=canonical_attack(),
                expected_detected=True,
            )
        ]
    )
    payload = result.as_dict()
    assert payload["total_cases"] == 1
    assert payload["precision"] == 1.0
    assert payload["recall"] == 1.0
