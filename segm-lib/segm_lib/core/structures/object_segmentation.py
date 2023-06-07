

class ObjectSegmentation:
	def __init__(self, classname: str, mask: dict, bbox: list):
		self.classname = classname
		self.mask = mask
		self.bbox = bbox