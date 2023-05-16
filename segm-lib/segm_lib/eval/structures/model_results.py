from dataclasses import dataclass, field, fields

from segm_lib.structures import Annotation, Prediction

@dataclass
class RawResults:
	n_images_with_predictions: int = 0
	n_objects_predicted: int = 0
	class_dist: dict[str, int] = field(default_factory=dict)

@dataclass
class EvalFilters:
	classes_considered: list[str] = field(default_factory=list)
	pred_classes_ignored: list[str] = field(default_factory=list)
	ann_classes_ignored: list[str] = field(default_factory=list)

@dataclass
class DatasetResults:
	n_anns_considered: int = 0
	n_preds_considered: int = 0

	mAP: float = 0.0
	n_true_positives: int = 0
	n_false_positives: int = 0
	n_false_negatives: int = 0

@dataclass
class ImageResults:
	n_anns_considered: int = 0
	n_preds_considered: int = 0

	mAP: float = 0.0
	true_positives: list[Prediction, Annotation] = field(default_factory=list)
	false_positives: list[Prediction] = field(default_factory=list)
	false_negatives: list[Annotation] = field(default_factory=list)
	
@dataclass
class ModelResults:
	raw_results: RawResults = None
	eval_filters: EvalFilters = None
	results_on_dataset: DatasetResults = None
	results_per_image: dict[str, ImageResults] = field(default_factory=dict)

	@classmethod
	def from_dict(cls, d: dict) -> "ModelResults":
		field_names = (field.name for field in fields(cls))
		return cls(**{k: v for k, v in d.items() if k in field_names})