from dataclasses import dataclass, field


@dataclass
class EvalFilters:
	evaluatable_classes: set[str] = field(default_factory=set)
	predicted_classes_impossible_to_evaluate: set[str] = field(default_factory=set)
	annotated_classes_impossible_to_predict: set[str] = field(default_factory=set)
	predictable_classes_impossible_to_evaluate: set[str] = field(default_factory=set)