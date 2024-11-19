import argparse

import cv2

from image_quality_metrics.quality_metrics import utils
from image_quality_metrics.quality_metrics.fft import fft_quality


def main(video, threshold):
	if video.isdigit():
		video = int(video)
	cap = cv2.VideoCapture(video)

	# Check if the webcam is opened correctly
	if not cap.isOpened():
		raise IOError("Cannot load video or open webcam")

	# loop over the frames from the video stream
	while True:
		# load the next frame
		ret, frame = cap.read()

		# If frame is read correctly ret is True
		if not ret:
			print("Error reading frame.")
			break

		frame = utils.resize(frame, size=512)

		quality_score, is_good_quality = fft_quality(frame, block_freq=60, t=threshold, to_gray=True, show=False)

		# annotate the image
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
	parser.add_argument("-t", "--threshold", type=float, default=12., help="If above the threshold, the image will be considered of good quality.")
	known_args, _ = parser.parse_known_args()

	main(
		known_args.video,
		known_args.threshold
	)