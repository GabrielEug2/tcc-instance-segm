from pathlib import Path
import json

import cv2

from . import common_logic

# # TODO update
# def plot(ann_dir: Path, out_dir: Path, save_masks: bool):
# 	ann_file = Path(args.ann_dir) / 'annotations.json'
# 	img_dir = Path(args.ann_dir) / 'images'
# 	img_files = img_dir.glob('*.jpg')

#     ann_dir = Path(ann_dir)
#     out_dir = Path(out_dir)
#     img_files = 
#     ann_file = 
    
#     anns = annotations.load(ann_file)
    
# 	# make classmap file
# 	# then read it

#     classmap_file = ann_file.parent / 'test_map.json'
#     metadata = common_logic.get_metadata(classmap_file)

#     if not out_dir.exists():
#         out_dir.mkdir()

#     for img_file in img_files:
#         formatted_anns = _get_annotations(anns, img_file)
#         if len(formatted_anns) == 0:
#             print(f"No annotations found on \"{str(ann_file)} for image \"{str(img_file)}\". Skipping")
#             continue
        
#         annotated_img = common_logic.plot(formatted_anns, img_file, metadata)
#         annotated_img_file = out_dir  / f"{img_file.stem}_groundtruth.jpg"
#         cv2.imwrite(str(annotated_img_file), annotated_img)

# def _load(ann_file):
#     ann_file = Path(ann_file)

#     with ann_file.open('r') as f:
#         anns = json.load(f)

#     return anns

# def _get_annotations(anns, img_file):
#     img_desc = _get_img_desc(anns, img_file)
#     if img_desc == None:
#         return []
    
#     relevant_anns = _filter_annotations(anns, img_desc['id'])
#     if len(relevant_anns) == 0:
#         return []

#     formatted_anns = _to_pred_format(relevant_anns, img_desc)
#     return formatted_anns

# def _get_img_desc(anns, img_file):
#     requested_img_desc = None
#     for img_desc in anns['images']:
#         if img_desc['file_name'] == img_file.name:
#             requested_img_desc = img_desc
#             break
#     return requested_img_desc

# def _filter_annotations(anns, img_id):
#     relevant_anns = []
#     for ann in anns['annotations']:
#         if ann['image_id'] == img_id:
#             relevant_anns.append(ann)
#     return relevant_anns

# def _to_pred_format(anns, img_desc):
#     formatted_anns = []

#     for ann in anns:
#         class_id = ann['category_id']
#         confidence = 100.0
#         mask = format_utils.ann_to_rle(ann['segmentation'], img_desc)
#         bbox = ann['bbox'] # tá em [x1,y1,h,w]
#         x1, y1, h, w = bbox[0], bbox[1], bbox[2], bbox[3]
#         x2 = x1 + w
#         y2 = y1 + h
#         bbox = [x1, y1, x2, y2]

#         pred = {
#             'class_id': class_id,
#             'confidence': confidence,
#             'mask': mask,
#             'bbox': bbox,
#         }
#         formatted_anns.append(pred)

#     return formatted_anns

# def _save_masks(predictions, out_dir):
# 	out_dir.mkdir(exist_ok=True)

# 	count_per_class = {}
# 	for pred in predictions:
# 		classname = pred['classname']
# 		mask = pred['mask']
		
# 		count_per_class[classname] = count_per_class.get(classname, 0) + 1
# 		mask_file = out_dir / f"{classname}_{count_per_class[classname]}.jpg"
# 		# TODO check if mask needs to be a specific type
# 		cv2.imwrite(str(mask_file), mask)