import argparse

def plot_annotations(img_file_or_dir, ann_file, out_dir):
	pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser("PLots annotations")
    parser.add_argument('-m', '--masks-too', help='whether or not to save individual masks as images',
                            action='store_true')
    parser.add_argument('img_file_or_dir', help='image or directory of images')
    parser.add_argument('ann_file', help='file containing the annotations for said image(s).')
    parser.add_argument('out_dir', help='directory to save the results')

    args = parser.parse_args()
    
	plot_annotations(args.img_file_or_dir, args.ann_file, args.out_dir)