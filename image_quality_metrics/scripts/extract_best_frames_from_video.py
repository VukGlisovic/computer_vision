"""
This script expects as input a path to video file. It
extracts the highest/lowest quality frames from the video
stream.
"""
import os
import argparse
import heapq
from typing import List
from collections.abc import Callable

import cv2
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

from image_quality_metrics.quality_metrics import utils
from image_quality_metrics.quality_metrics.fft import fft_quality


def find_local_extrema(data: List[float], comparator: Callable = np.greater, order: int = 10) -> List[int]:
	"""Finds the indices of local extrema based on the comparator.
	"""
	local_maxima_indices = argrelextrema(np.array(data), comparator=comparator, order=order)[0]
	return local_maxima_indices.tolist()


def get_top_n_indices(data: List, n: int, higher_is_better: bool = True) -> List[int]:
	"""Gets the indices of the highest/lowest values in the data.
	"""
	method = heapq.nlargest if higher_is_better else heapq.nsmallest
	return [index for index, _ in method(n, enumerate(data), key=lambda x: x[1])]


def save_frames(frames: List[np.ndarray], indices: List[int], output_dir: str, higher_is_better: bool = True) -> None:
	"""Extracts the best frames and stores them to disk.
	"""
	best_worst = 'best' if higher_is_better else 'worst'
	for i in indices:
		cv2.imwrite(os.path.join(output_dir, f'{best_worst}_frame_{i:04d}.jpg'), frames[i])


def plot_quality_scores(quality_scores: List[float], output_dir: str, top_indices: list = None, worst_indices: list = None) -> None:
	"""Plots a timeseries of the quality scores and saves the figure to disk.
	"""
	fig, ax = plt.subplots(figsize=(15, 4))
	ax.plot(range(len(quality_scores)), quality_scores)
	if top_indices:
		ax.plot(top_indices, [quality_scores[i] for i in top_indices], 'x', color='green')
	if worst_indices:
		ax.plot(worst_indices, [quality_scores[i] for i in worst_indices], 'x', color='red')
	ax.grid(lw=0.5, ls='--', alpha=0.5)
	ax.set_xlabel('frame number', fontsize=14)
	ax.set_ylabel('quality score', fontsize=14)
	plt.savefig(os.path.join(output_dir, 'quality_scores.jpg'))


def main(video: str, nr_frames: int, block_freq: int, output_dir: str) -> None:
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
			# Break at the end of the video
			break

		# Calculate quality and annotate frame
		frame = utils.resize(frame, size=512)
		quality_score = fft_quality(frame, block_freq=block_freq, to_gray=True, show=False)
		frame = utils.annotate_quality_result(frame, quality_score, is_good_quality=True)

		# Save results
		frames.append(frame)
		quality_scores.append(quality_score)

	cap.release()

	# Save best/worst frames and plot quality score timeseries
	os.makedirs(output_dir, exist_ok=True)
	# find local maxima for best frames
	best_indices = find_local_extrema(quality_scores, comparator=np.greater, order=12)
	save_frames(frames, best_indices, output_dir, higher_is_better=True)
	# find minima for worst frames
	worst_indices = get_top_n_indices(quality_scores, nr_frames, higher_is_better=False)
	save_frames(frames, worst_indices, output_dir, higher_is_better=False)
	plot_quality_scores(quality_scores, output_dir, best_indices, worst_indices)
	print("Finished extracting frames!")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-v", "--video", type=str, help="Specify a path to a video file.")
	parser.add_argument("-n", "--nr_frames", type=int, default=10, help="Number of frames to extract.")
	parser.add_argument("-b", "--block_freq", type=int, default=20, help="The number of lower frequencies to block.")
	parser.add_argument("-o", "--output_dir", type=str, default="extracted_frames/", help="Where to store the extracted frames.")
	known_args, _ = parser.parse_known_args()

	main(
		known_args.video,
		known_args.nr_frames,
		known_args.block_freq,
		known_args.output_dir
	)
