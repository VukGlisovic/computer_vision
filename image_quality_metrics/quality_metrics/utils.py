import cv2


def resize(img, size):
	"""Resizes while preserving aspect ratio.

	Args:
		img (np.ndarray):
		size (int):

	Returns:
		np.ndarray
	"""
	h, w, _ = img.shape
	r = max(h / size, w / size)
	new_h, new_w = int(round(h / r)), int(round(w / r))
	img = cv2.resize(img, dsize=(new_w, new_h))
	return img


def annotate_quality_result(img, quality_score, is_good_quality):
	# color is in BGR format -> (0, 255, 0)=green, (0, 0, 255)=red
	color = (0, 255, 0) if is_good_quality else (0, 0, 255)
	text = f"Good quality ({quality_score:.4f})" if is_good_quality else f"Bad quality ({quality_score:.4f})"
	cv2.putText(img, text, (10, 25), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=color, thickness=1)
	return img
