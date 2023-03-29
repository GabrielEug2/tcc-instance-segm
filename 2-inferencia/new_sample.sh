#!/bin/bash
IMG_DIR="$HOME/tcc/data/openimages-200/images"

if [[ -n $1 && -n $2 ]]; then
    n_imgs=$1
    out_dir=$2
else
    echo "Usage: new_sample.sh <n_imgs> <out_dir"
    exit 1
fi

mapfile -t random_imgs < <(find "$IMG_DIR" | sort -R | tail -n "$n_imgs")

rm -rf "$out_dir"
mkdir -p "$out_dir" && cp "${random_imgs[@]}" "$out_dir"