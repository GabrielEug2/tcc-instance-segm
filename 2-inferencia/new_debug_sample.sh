#!/bin/bash
img_dir="~/tcc/data/openim_200/images"
sample_img_dir="tmp"
sample_out_dir="tmp_output"

mapfile -t imgs_names < <(find "$img_dir" | sort -R | tail -n 3)

img_list=""
for i in "${imgs_names[@]}"
do
    img_list+="$i "
done

rm -rf "$sample_img_dir"
rm -rf "$sample_out_dir"
mkdir -p "$sample_img_dir" && cp $img_list "$sample_img_dir"