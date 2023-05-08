
# for model get results for img x
	# concat save
	# result_dict = {
	# 	'mAP_on_all_images': {},
	# 	'mAP_per_image': {},
	# }
	# for model_name, results in results_per_model.items():
	# 	result_dict['mAP_on_all_images'][model_name] = results.ap_on_all_imgs
		
	# 	for img, ap_for_image in results.ap_per_image.items():
	# 		if img not in result_dict['mAP_per_image']:
	# 			result_dict['mAP_per_image'][img] = {}

	# 		result_dict['mAP_per_image'][img][model_name] = ap_for_image

	# results_file = out_dir / 'results.json'
	# with results_file.open('w') as f:
	# 	json.dump(result_dict, f)