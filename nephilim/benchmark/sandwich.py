"""Evidence-first benchmark harness for the deterministic sandwich detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from nephilim.classifier.sandwich_detector import SandwichDetector
from nephilim.stream.transaction_decoder import TxRecord


@dataclass(frozen=True)
class SandwichBenchmarkCase:
    """One labeled block/case for detector evaluation."""

    name: str
    txs: list[TxRecord]
    expected_detected: bool


@dataclass(frozen=True)
class SandwichBenchmarkResult:
    total_cases: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    false_positive_rate: float
    accuracy: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "total_cases": self.total_cases,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "false_positive_rate": self.false_positive_rate,
            "accuracy": self.accuracy,
        }


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def evaluate_sandwich_detector(
    cases: Iterable[SandwichBenchmarkCase],
    detector: SandwichDetector | None = None,
) -> SandwichBenchmarkResult:
    """Evaluate case-level detection metrics on a labeled corpus.

    This harness intentionally makes no claim that the corpus is representative.
    Synthetic/adversarial fixtures are useful for regression. Historical-chain
    validation requires a separately sourced and labeled corpus.
    """
    detector = detector or SandwichDetector()
    tp = fp = tn = fn = 0
    count = 0

    for case in cases:
        count += 1
        detected = bool(detector.detect(case.txs))
        if detected and case.expected_detected:
            tp += 1
        elif detected and not case.expected_detected:
            fp += 1
        elif not detected and not case.expected_detected:
            tn += 1
        else:
            fn += 1

    return SandwichBenchmarkResult(
        total_cases=count,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=_ratio(tp, tp + fp),
        recall=_ratio(tp, tp + fn),
        false_positive_rate=_ratio(fp, fp + tn),
        accuracy=_ratio(tp + tn, count),
    )
