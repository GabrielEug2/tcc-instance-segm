#!/bin/bash

inference_script=/home/gabriel/tcc/segmentation-tools/2-inferencia/inference.py
img_dir=~/tcc/data/openim_200/images
out_dir=~/tcc/data/openim_200_output

rm -rf $out_dir
python $inference_script $img_dir $out_dir