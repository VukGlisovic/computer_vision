"""
This script expects as input a path to video file. It
extracts the highest quality frames from the video stream.
"""
import os
import argparse
import heapq

import cv2
import matplotlib.pyplot as plt

from image_quality_metrics.quality_metrics import utils
from image_quality_metrics.quality_metrics.fft import fft_quality


def get_top_n_indices(data: list, n: int) -> list:
	return [index for index, _ in heapq.nlargest(n, enumerate(data), key=lambda x: x[1])]


def plot_quality_scores(quality_scores, output_dir):
	fig, ax = plt.subplots(figsize=(15, 4))
	ax.plot(range(len(quality_scores)), quality_scores)
	ax.grid(lw=0.5, ls='--', alpha=0.5)
	ax.set_xlabel('frame number', fontsize=14)
	ax.set_ylabel('quality score', fontsize=14)
	plt.savefig(os.path.join(output_dir, 'quality_scores.jpg'))


def main(video: str, nr_frames: int, output_dir: str) -> None:
	cap = cv2.VideoCapture(video)

	# Check if the webcam is opened correctly
	if not cap.isOpened():
		raise IOError("Cannot load video.")

	# Loop over the frames from the video stream
	frames = []
	quality_scores = []
	while True:
		# Load the next frame
		ret, frame = cap.read()

		# If the frame is read correctly ret is True
		if not ret:
			break

		# calculate quality and annotate frame
		frame = utils.resize(frame, size=512)
		quality_score = fft_quality(frame, block_freq=60, to_gray=True, show=False)
		frame = utils.annotate_quality_result(frame, quality_score, is_good_quality=True)

		# save results
		frames.append(frame)
		quality_scores.append(quality_score)

	cap.release()

	# save best frames and plot quality score timeseries
	os.makedirs(output_dir, exist_ok=True)
	top_indices = get_top_n_indices(quality_scores, nr_frames)
	for i in top_indices:
		cv2.imwrite(os.path.join(output_dir, f'frame_{i:04d}.jpg'), frames[i])
	plot_quality_scores(quality_scores, output_dir)
	print("Finished extracting frames!")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-v", "--video", type=str, help="Specify a path to a video file.")
	parser.add_argument("-n", "--nr_frames", type=int, default=5, help="Number of frames to extract.")
	parser.add_argument("-o", "--output_dir", type=str, default="extracted_frames/", help="Where to store the extracted frames.")
	known_args, _ = parser.parse_known_args()

	main(
		known_args.video,
		known_args.nr_frames,
		known_args.output_dir
	)
