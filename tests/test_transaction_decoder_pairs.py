from nephilim.stream.transaction_decoder import (
    _decode_v2_path_pair,
    _decode_v3_exact_input_single_pair,
    _UNI_V2_SWAP_ETH_FOR_TOKENS,
)


def _word_address(address: str) -> str:
    return "0" * 24 + address.lower().removeprefix("0x")


def _word_int(value: int) -> str:
    return f"{value:064x}"


def test_decode_v3_exact_input_single_pair():
    token_a = "0x1111111111111111111111111111111111111111"
    token_b = "0x2222222222222222222222222222222222222222"
    calldata = (
        "0x414bf389"
        + _word_address(token_a)
        + _word_address(token_b)
        + _word_int(3000)
        + _word_address("0x3333333333333333333333333333333333333333")
        + _word_int(0) * 4
    )
    key = _decode_v3_exact_input_single_pair(calldata)
    assert key == f"uniswap_v3:{token_a}:{token_b}:3000"


def test_decode_v2_path_pair():
    token_a = "0x1111111111111111111111111111111111111111"
    token_b = "0x2222222222222222222222222222222222222222"
    # amountOutMin, path offset=128 bytes, to, deadline, then dynamic path.
    calldata = (
        _UNI_V2_SWAP_ETH_FOR_TOKENS
        + _word_int(1)
        + _word_int(128)
        + _word_address("0x3333333333333333333333333333333333333333")
        + _word_int(999999)
        + _word_int(2)
        + _word_address(token_a)
        + _word_address(token_b)
    )
    key = _decode_v2_path_pair(calldata, _UNI_V2_SWAP_ETH_FOR_TOKENS)
    assert key == f"uniswap_v2:{token_a}:{token_b}"
