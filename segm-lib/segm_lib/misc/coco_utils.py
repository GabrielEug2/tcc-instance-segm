import json
from collections import defaultdict
from pathlib import Path

from ..core.managers.coco_ann_manager import COCOAnnManager

EXPECTED_FILENAMES = {
	'train': 'instances_train2017.json',
	'val': 'instances_val2017.json'
}

def class_dist(coco_ann_dir: Path, out_file: Path, verbose: bool = True):
	"""Computes the class distribution for the COCO dataset.

	Args:
		coco_ann_dir (Path): dir where the annotation files are.
		out_file (Path): file to save the results.

	Raises:
		FileNotFoundError: if none of the expected files were found on coco_ann_dir.
	"""
	coco_ann_files = [coco_ann_dir / fn for fn in EXPECTED_FILENAMES.values()]
	if all(not f.exists() for f in coco_ann_files):
		raise FileNotFoundError(f'None of the expected files {EXPECTED_FILENAMES} '
		                        f'were found on dir "{str(coco_ann_dir)}"')

	total_class_dist = defaultdict(lambda: 0)
	for ann_file in coco_ann_files:
		if verbose:
			# Só pra saber se é normal a demora, tipo o instances_train.2017
			print(f'  Processing {ann_file.name}...', end='')

		file_dist = COCOAnnManager(ann_file).class_distribution()
		for classname in file_dist:
			total_class_dist[classname] += file_dist[classname]

		if verbose:
			print('done')

	dist_sorted_by_count = dict(sorted(
		total_class_dist.items(),
		key=lambda c: c[1],
		reverse=True
	))
	out_file.parent.mkdir(parents=True, exist_ok=True)
	with out_file.open('w') as f:
		json.dump(dist_sorted_by_count, f, indent=4)

def class_map(coco_ann_dir: Path, out_file: Path):
	"""Computes the class map for the COCO dataset.

	Args:
		coco_ann_dir (Path): dir where the annotation files are.
		out_file (Path): file to save the results.

	Raises:
		FileNotFoundError: if none of the expected files were found on coco_ann_dir.
		ValueError: if a file does not follow the expected format.
	"""
	# Não importa qual arquivo (train ou val), os dois tem as mesmas classes.
	# Eu só uso o val porque ele é menor / carrega mais rápido.
	coco_ann_file = coco_ann_dir / EXPECTED_FILENAMES['val']
	if not coco_ann_file.exists():
		raise FileNotFoundError(f'Expected file "{coco_ann_file}" not found '
		                        f'on dir "{str(coco_ann_dir)}"')
	
	# O COCO pula alguns IDs: tem 80 classes, mas vai até o ID ~90.
	# Para simplificar, os modelos normalizam pra [0,N)
	coco_anns = COCOAnnManager(coco_ann_file)
	default_map = coco_anns.classmap()
	normalized_map = coco_anns.normalized_classmap()
	classmaps = { 'default': default_map, 'normalized': normalized_map }

	out_dir = out_file.parent
	out_file_basename = out_file.stem
	out_file_extension = out_file.suffix # already has the trailing '.'
	out_dir.mkdir(parents=True, exist_ok=True)
	for map_name, classmap in classmaps.items():
		map_file = out_dir / f"{out_file_basename}_{map_name}{out_file_extension}"
		with map_file.open('w') as f:
			json.dump(classmap, f, indent=4)