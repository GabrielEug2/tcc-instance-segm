from dataclasses import dataclass, field

@dataclass
class ImageInfo:
	n_objects: int = 0
	class_dist: dict[str, int] = field(default_factory=dict)

@dataclass
class DatasetInfo:
	n_images: int = 0
	n_objects: int = 0
	class_dist: dict[str, int] = field(default_factory=dict)

	info_per_image: dict[str, ImageInfo] = field(default_factory=dict)