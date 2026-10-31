"""Estimated compute (primary metric) and dollar cost (secondary metric).
Both always use the token counts returned by the provider, never estimates."""


def call_compute_flops(info: dict, input_tokens: int, output_tokens: int) -> float:
    """Estimated compute of one call: 2 × active parameters × all tokens (FLOPs, estimate)."""
    return 2 * info["active_params_b"] * 1e9 * (input_tokens + output_tokens)


def call_cost_usd(info: dict, input_tokens: int, output_tokens: int) -> float | None:
    """Dollar cost, or None when the model's price is not verified."""
    if not info.get("price_verified"):
        return None
    return (input_tokens * info["input_price"] + output_tokens * info["output_price"]) / 1_000_000


def per_success(values: list[float | None], n_correct: int) -> float | None:
    """Total over all calls (including failed attempts and escalations) / number correct.
    None when nothing is correct, or when any value is None (e.g. an unverified price)."""
    if n_correct == 0 or any(v is None for v in values):
        return None
    return sum(values) / n_correct
