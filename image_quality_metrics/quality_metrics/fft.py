import cv2
import numpy as np
import matplotlib.pyplot as plt


def spectral_entropy(fft: np.ndarray) -> float:
	"""Calculates the spectral entropy of an image.
	"""
	magnitude_spectrum = np.abs(fft)  # calculates the magnitude via sqrt(real^2 + imag^2)
	power_spectrum = magnitude_spectrum ** 2  # calculate power spectral density
	normalized_ps = power_spectrum / np.sum(power_spectrum)  # normalize sothat it can be viewed as a probability density
	entropy = -np.sum(normalized_ps * np.log2(normalized_ps + 1e-10))  # Add a small value to avoid log(0)
	return entropy


def fft_quality(img: np.ndarray, block_freq: int = 60, to_gray: bool = True, show: bool = False) -> float:
	"""Calculates the spectral entropy of the input img.

	Args:
		img:
		block_freq: which frequencies in the fft to block from spectral
			entropy calculations.
		to_gray: whether the spectrum entropy calculation should be done
			on the grayscale image or the RGB image.
		show: if True, then plots the input image, the magnitude spectrum
			and the reconstructed image.

	Returns:
		The spectral entropy value
	"""
	if to_gray:
		# convert RGB image to grayscale image
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = np.expand_dims(img, axis=-1)  # add channel dimension

	# fft2 outputs an array of the same size as the input img of complex numbers.
	# Each complex number has two components:
	# magnitude -> represents the amplitude/strength of a frequency
	# phase -> represents the frequency or offset of that component
	fft = np.fft.fft2(img, axes=[0, 1])

	# zero-out the center of the FFT shift (i.e., remove low
	# frequencies), apply the inverse shift such that the DC
	# component once again becomes the top-left, and then apply
	# the inverse FFT
	h, w, _ = img.shape
	center_x, center_y = w // 2, h // 2
	fft_shifted_filtered = np.fft.fftshift(fft, axes=[0, 1])  # shift the frequencies
	fft_shifted_filtered[center_y - block_freq: center_y + block_freq, center_x - block_freq: center_x + block_freq] = 0
	fft_filtered = np.fft.ifftshift(fft_shifted_filtered, axes=[0, 1])

	if show:
		visualize_fft_images(img, fft_shifted_filtered, fft_filtered)

	entropy = spectral_entropy(fft_filtered)

	return entropy


def visualize_fft_images(img: np.ndarray, fft_shifted_filtered: np.ndarray, fft_filtered: np.ndarray) -> None:
	"""Plots the input image, the magnitude spectrum and the reconstructed image.
	"""
	# use fft_filtered_shifted for magnitude calculation
	magnitude_spectrum = np.sqrt(np.abs(fft_shifted_filtered))
	magnitude_spectrum = magnitude_spectrum.mean(axis=-1)
	# use fft_filtered for image reconstruction
	img_reconstructed = np.fft.ifft2(fft_filtered, axes=[0, 1])  # reconstruct based on high frequencies only

	fig, axes = plt.subplots(1, 3, figsize=(21, 7))
	ax1, ax2, ax3 = axes
	# show the original grayscale input image
	ax1.imshow(img[:,:,[2,1,0]])
	ax1.set_title("Input image", fontsize=14)
	# show the log magnitude image
	ax2.imshow(magnitude_spectrum, cmap="gray")
	ax2.set_title("Magnitude Spectrum", fontsize=14)
	# plot the reconstructed image
	img_reconstructed = (img_reconstructed.real - img_reconstructed.real.min()) / (img_reconstructed.real.max() - img_reconstructed.real.min())
	# img_reconstructed = img_reconstructed.real
	if img_reconstructed.shape[-1] == 3:
		# matplotlib expects RGB images
		img_reconstructed = img_reconstructed[:,:,[2,1,0]]
	ax3.imshow(img_reconstructed)
	ax3.set_title("Reconstructed image", fontsize=14)
	# make the figure pretty
	for ax in axes:
		ax.set_xticks([])
		ax.set_yticks([])
	# show our plots
	plt.show()
