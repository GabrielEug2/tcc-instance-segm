from pathlib import Path

from tqdm import tqdm

from segm_lib.core.managers import AnnManager
from segm_lib.core.managers import MultiModelPredManager, SingleModelPredManager
from segm_lib.plot import PlotLib

VALID_LIBRARIES = ['det', 'bin']
LIBRARY_MAP: dict[str, type[PlotLib]] = {}

def plot_annotations(ann_dir: Path, img_dir: Path, out_dir: Path, libs: list[str] = None):
	"""Plots the annotations.

	Args:
		ann_dir (Path): directory containing the annotations
		img_dir (Path): directory containing the images the annotations refer to
		out_dir (Path): directory to save the results
		libs (list[str], optional): list of libs to use. See VALID_LIBRARIES in
			this same module for a list of valid libraries. By default, uses all
			of the available.
	"""
	requested_libs = VALID_LIBRARIES if libs is None else libs
	try:
		_load_libs(requested_libs)
	except:
		raise

	ann_manager = AnnManager(ann_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	print(f'Plotting annotations with libraries {requested_libs}...')

	for lib_name in requested_libs:
		print(f'\n{lib_name} lib...')
		lib_instance = LIBRARY_MAP[lib_name]()

		_plot_everything(ann_manager, img_dir, lib_instance, out_dir, 'groundtruth', lib_name)


def plot_predictions(pred_dir: Path, img_dir: Path, out_dir: Path, libs: list[str] = None):
	"""Plots the predictions.

	Args:
		pred_dir (Path): directory containing the predictions
		img_dir (Path): directory containing the images the predictions refer to
		out_dir (Path): directory to save the results
		libs (list[str], optional): list of libs to use. See VALID_LIBRARIES in
			this same module for a list of valid libraries. By default, uses all
			of the available.
	"""
	requested_libs = VALID_LIBRARIES if libs is None else libs
	try:
		_load_libs(requested_libs)
	except:
		raise

	pred_manager = MultiModelPredManager(pred_dir)

	model_names = pred_manager.get_model_names()
	if len(model_names) == 0:
		print(f'  No predictions found on {pred_dir}.')
		return
	print(f'  Found predictions for models {model_names}')

	out_dir.mkdir(parents=True, exist_ok=True)
	print(f'Plotting with libraries {requested_libs}...')

	for lib_name in requested_libs:
		print(f'{lib_name} lib...')
		lib_instance = LIBRARY_MAP[lib_name]()

		for model_name in model_names:
			model_pred_manager = pred_manager.get_manager(model_name)
			print(f'\n  {model_name}...')

			_plot_everything(model_pred_manager, img_dir, lib_instance, out_dir, model_name, lib_name)

def _load_libs(libs):
	for lib_name in libs:
		try:
			_import_plot_lib(lib_name)
		except ImportError as e:
			raise ImportError(f'"{lib_name}" is implemented on the library, but not installed properly.') from e
		except ValueError:
			raise ValueError(f'"{lib_name}" is not implemented/setup correctly. How do you expect to run that.')
		
def _import_plot_lib(lib_name):
	# Imports are conditional because you don't need
	# to install all the libs
	match lib_name:
		case 'det':
			from segm_lib.plot.detectron_plot_lib import DetectronPlotLib as class_ref
		case 'bin':
			from segm_lib.plot.bin_plot_lib import BinPlotLib as class_ref
		case other:
			raise ValueError()

	LIBRARY_MAP[lib_name] = class_ref

def _plot_everything(
		ann_or_pred_manager: AnnManager|SingleModelPredManager,
		img_dir: Path,
		plot_lib: PlotLib,
		out_dir: Path,
		out_basename: str,
		lib_name: str,
		):
	img_files = img_dir.glob('*.jpg')
	n_images = sum(1 for _ in img_dir.glob('*.jpg'))

	for img_file in tqdm(img_files, total=n_images):
		anns_or_preds = ann_or_pred_manager.load(img_file.stem)
		if len(anns_or_preds) == 0:
			continue

		plot_img_file = out_dir / img_file.stem / lib_name / f'{out_basename}.jpg'
		plot_img_file.parent.mkdir(parents=True, exist_ok=True)
		plot_lib.plot(anns_or_preds, img_file, plot_img_file)

		masks_dir = out_dir / img_file.stem / lib_name / f'{out_basename}_masks'
		masks_dir.mkdir(parents=True, exist_ok=True)
		plot_lib.plot_individual_masks(anns_or_preds, img_file, masks_dir)