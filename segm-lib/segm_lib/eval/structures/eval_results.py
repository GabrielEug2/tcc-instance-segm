from dataclasses import dataclass, field
from typing import Any


@dataclass(kw_only=True)
class EvalResults:
	n_anns_considered: int = 0
	class_dist_anns_considered: dict[str, int] = field(default_factory=dict)
	n_preds_considered: int = 0
	class_dist_preds_considered: dict[str, int] = field(default_factory=dict)

	AP: float = 0
	true_positives: dict[str, Any] = field(default_factory=dict)
	false_positives: dict[str, Any] = field(default_factory=dict)
	false_negatives: dict[str, Any] = field(default_factory=dict)