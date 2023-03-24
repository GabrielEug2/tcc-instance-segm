
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog

# import cv2

# “instances”: Instances object with the following fields:
#     “pred_boxes”: Boxes object storing N boxes, one for each detected instance.
#     “scores”: Tensor, a vector of N confidence scores.
#     “pred_classes”: Tensor, a vector of N labels in range [0, num_categories).
#     “pred_masks”: a Tensor of shape (N, H, W), masks for each detected instance.
#     “pred_keypoints”: a Tensor of shape (N, num_keypoint, 3). Each row in the last dimension is (x, y, score). Confidence scores are larger than 0.

# out_img = plot(instances, img)

# def plot(instances, img):
#     # o Metadata é pra saber o nome das classes
#     v = Visualizer(img, MetadataCatalog.get('coco_2017_test'), scale=1.2)
#     vis_out = v.draw_instance_predictions(instances)

#     return vis_out.get_image()


# img_id, extension = os.path.basename(img_path).split('.')
# out_filename = f"{img_id}_{model_name}.{extension}"

# cv2.imwrite(os.path.join(args.output_dir, out_filename), out_img)
