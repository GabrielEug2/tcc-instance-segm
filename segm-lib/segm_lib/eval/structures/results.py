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
	anns_considered: 'AnnsOrPredsInfo' = field(default_factory=lambda: AnnsOrPredsInfo())
	preds_considered: 'AnnsOrPredsInfo' = field(default_factory=lambda: AnnsOrPredsInfo())
	AP: float = 0.0

@dataclass(kw_only=True)
class DatasetResults(CommonResults):
	true_positives: 'TP_FP_FN_ShortInfo' = field(default_factory=lambda: TP_FP_FN_ShortInfo())
	false_positives: 'TP_FP_FN_ShortInfo' = field(default_factory=lambda: TP_FP_FN_ShortInfo())
	false_negatives: 'TP_FP_FN_ShortInfo' = field(default_factory=lambda: TP_FP_FN_ShortInfo())

@dataclass(kw_only=True)
class ImgResults(CommonResults):
	true_positives: 'TP_FP_FN_DetailedInfo' = field(default_factory=lambda: TP_FP_FN_DetailedInfo())
	false_positives: 'TP_FP_FN_DetailedInfo' = field(default_factory=lambda: TP_FP_FN_DetailedInfo())
	false_negatives: 'TP_FP_FN_DetailedInfo' = field(default_factory=lambda: TP_FP_FN_DetailedInfo())

@dataclass
class AnnsOrPredsInfo:
	n: int = 0
	class_dist: dict[str, int] = field(default_factory=dict)

@dataclass
class TP_FP_FN_ShortInfo:
	n: int = 0
	n_per_class: dict[str, int] = field(default_factory=dict)

@dataclass
class TP_FP_FN_DetailedInfo(TP_FP_FN_ShortInfo):
	list_per_class: dict[str, list] = field(default_factory=dict)