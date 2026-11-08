"""Shared constants for bracket loading, merging and tonemapping."""

import cv2
import numpy as np

# Formats holding pixels a camera has already curved, against formats holding the sensor's own
# readings, which is the difference that decides how a bracket has to be merged.
ENCODED_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
RAW_SUFFIXES = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw"}
IMAGE_SUFFIXES = ENCODED_SUFFIXES | RAW_SUFFIXES

# Readings this far up the range stopped climbing before the shutter closed.
SATURATION = 0.99
# The range an encoded shot is held in, and the reading at the top of it, which a response curve
# is indexed by and a merge weighs the least.
LEVELS = 256
SATURATED = LEVELS - 1

# The curve that stands between how much light a pixel caught and how bright it looks.
DISPLAY_GAMMA = 2.2

# Each calibrator recovers the camera response curve that its matching merger expects.
CALIBRATORS = {
    "debevec": cv2.createCalibrateDebevec,
    "robertson": cv2.createCalibrateRobertson,
}
MERGERS = {
    "debevec": cv2.createMergeDebevec,
    "robertson": cv2.createMergeRobertson,
}

# Rec.709 luminance weights, in the blue to red order that the images come in.
LUMINANCE_WEIGHTS = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)

# The two matrices Stephen Hill fits the ACES rendering around, carrying linear sRGB into the
# space the curve is shaped for and back out again.
ACES_INPUT = np.array([[0.59719, 0.35458, 0.04823],
                       [0.07600, 0.90834, 0.01566],
                       [0.02840, 0.13383, 0.83777]], dtype=np.float32)
ACES_OUTPUT = np.array([[1.60475, -0.53108, -0.07367],
                        [-0.10208, 1.10813, -0.00605],
                        [-0.00327, -0.07276, 1.07602]], dtype=np.float32)
# Where a tonemapping operator puts mid grey, which the ACES curve is drawn around.
MIDDLE_GREY = 0.18

# How far apart the darkest and brightest parts of a scene are left once the broad shape of it
# has been flattened, as a ratio between the two.
BASE_CONTRAST = 5.0
# How far apart two readings have to be, in decades, for an edge to be taken to run between them.
BASE_EDGE = 0.4
