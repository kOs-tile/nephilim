# Validation plan

NEPHILIM is an evidence-first MEV detector research lab. Synthetic fixtures are
for regression, not for production accuracy claims.

## Primary safety metric

**False-positive rate** on independently labeled historical blocks.

## Corpus progression

### Stage 0 — deterministic adversarial fixtures

Already covered in CI:
- canonical directional sandwich
- same-direction attacker traffic
- missing direction evidence
- equal-gas ordering
- wrong pair
- wrong backrun sender

### Stage 1 — historical labeled corpus

Create a versioned corpus with:
- transaction hashes and block numbers
- decoded calldata
- token direction
- trace references
- label source/reviewer
- ambiguous/unknown label support
- known positives and hard negatives

### Stage 2 — trace-level economics

Replace heuristic extracted-value estimates with trace/state-diff based realized
profit reconstruction.

## Metrics

- precision
- recall
- false-positive rate
- false-negative rate
- unsupported-calldata rate
- ambiguous-case rate
- pair/direction decode coverage
- trace-profit reconstruction error

## Exit gate

No production/trading-signal claim until historical labels and trace-level
profit evidence are reproducible. Synthetic 100% benchmark scores remain
regression evidence only.
