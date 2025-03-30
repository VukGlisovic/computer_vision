"""
This script expects as input a path to a video file or a
webcam stream, and it outputs a stream of images (video
frames) annotated with text describing the quality of each
frame.
"""
import argparse

import cv2

from image_quality_metrics.quality_metrics import utils
from image_quality_metrics.quality_metrics.fft import fft_quality


def main(video: str, block_freq: int, threshold: float):
	if video.isdigit():
		video = int(video)
	cap = cv2.VideoCapture(video)

	# Check if the video or webcam is opened correctly
	if not cap.isOpened():
		raise IOError("Cannot load video or open webcam")

	# Loop over the frames from the video stream
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
		is_good_quality = (quality_score >= threshold)
		frame = utils.annotate_quality_result(frame, quality_score, is_good_quality)

		# show the output frame
		cv2.imshow("Frame", frame)
		# Break the loop if 'q' key is pressed
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-v", "--video", type=str, default='0', help="Specify a path to a video file or point to a webcam. Usually 0 is the webcam.")
	parser.add_argument("-b", "--block_freq", type=int, default=20, help="The number of lower frequencies to block.")
	parser.add_argument("-t", "--threshold", type=float, default=12., help="If above the threshold, the image will be considered of good quality.")
	known_args, _ = parser.parse_known_args()

	main(
		known_args.video,
		known_args.block_freq,
		known_args.threshold
	)