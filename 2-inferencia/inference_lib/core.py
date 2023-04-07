
import inference_lib.inference
import inference_lib.view

def inference(img_file_or_dir, out_dir):
	"""Runs inference on the requested imgs.

	Args:
		img_file_or_dir (str): path to img or dir of images to segment.
		out_dir (str): directory to save the outputs.
	"""
	inference_lib.inference()
	inference_lib.view()