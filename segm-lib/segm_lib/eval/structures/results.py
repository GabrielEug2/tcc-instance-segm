from dataclasses import dataclass, field


@dataclass
class RawResults:
	n_images_with_predictions: int = 0
	n_objects_predicted: int = 0
	class_dist_for_predictions: dict[str, int] = field(default_factory=dict)

@dataclass
class EvalFilters:
	classes_considered: list[str] = field(default_factory=list)
	pred_classes_ignored: list[str] = field(default_factory=list)
	ann_classes_ignored: list[str] = field(default_factory=list)

@dataclass
class CommonResults:
	anns_considered: 'AnnsOrPredsInfo' = None
	preds_considered: 'AnnsOrPredsInfo' = None
	AP: float = 0.0

@dataclass(kw_only=True)
class DatasetResults(CommonResults):
	true_positives: 'TP_FP_FN_ShortInfo' = None
	false_positives: 'TP_FP_FN_ShortInfo' = None
	false_negatives: 'TP_FP_FN_ShortInfo' = None

@dataclass(kw_only=True)
class ImgResults(CommonResults):
	true_positives: 'TP_FP_FN_DetailedInfo' = None
	false_positives: 'TP_FP_FN_DetailedInfo' = None
	false_negatives: 'TP_FP_FN_DetailedInfo' = None

@dataclass
class AnnsOrPredsInfo:
	n: int = 0
	class_dist: dict[str, int] = field(default_factory=dict)

@dataclass
class TP_FP_FN_ShortInfo:
	n: int
	n_per_class: dict[str, int] = field(default_factory=dict)

@dataclass
class TP_FP_FN_DetailedInfo(TP_FP_FN_ShortInfo):
	list_per_class: dict[str, list] = field(default_factory=dict)