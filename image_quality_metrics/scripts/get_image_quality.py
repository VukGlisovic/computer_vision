import argparse

import cv2

from image_quality_metrics.quality_metrics import utils
from image_quality_metrics.quality_metrics.fft import fft_quality


def main(path, threshold):
	# load the image for which to check the quality
	img = cv2.imread(path)
	img = utils.resize(img, size=512)

	# calculate the quality score
	quality_score, is_good_quality = fft_quality(img, block_freq=40, t=threshold, to_gray=False, show=True)

	# annotate the image
	img = utils.annotate_quality_result(img, quality_score, is_good_quality)

	# show the annotated image
	cv2.imshow("Result", img)
	cv2.waitKey(0)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input_path", type=str, required=True, help="Path to input image.")
	parser.add_argument("-t", "--threshold", type=float, default=12., help="If above the threshold, the image will be considered of good quality.")
	known_args, _ = parser.parse_known_args()

	main(
		known_args.input_path,
		known_args.threshold
	)