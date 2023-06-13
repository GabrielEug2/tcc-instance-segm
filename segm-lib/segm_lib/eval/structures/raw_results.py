from dataclasses import dataclass, field


@dataclass
class RawResults:
	n_images_with_preds: int = 0
	n_objects_predicted: int = 0
	class_dist_on_preds: dict[str, int] = field(default_factory=dict)