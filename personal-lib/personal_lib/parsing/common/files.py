import json
from pathlib import Path

def save_class_dist(class_dist: dict, out_file: Path):
	dist_sorted_by_count = dict(sorted(class_dist.items(), key=lambda c: c[1], reverse=True))

	out_file.parent.mkdir(parents=True, exist_ok=True)
	with out_file.open('w') as f:
		json.dump(dist_sorted_by_count, f, indent=4)