"""Alignment and display encoding of bracket shots."""

import cv2
import numpy as np

from aeb.constants import DISPLAY_GAMMA


def downscale(images: list[np.ndarray], max_side: int | None) -> list[np.ndarray]:
    """Shrinks the shots until their longer side stays within max_side, or hands them back.

    Every step over a bracket costs with its pixel count, so a side limit trades resolution
    for a run that finishes sooner on big shots. All of the shots are shrunk to the one size,
    which keeps them on a common grid, and area averaging keeps what it folds together from
    aliasing into noise.
    """
    if not max_side or max_side <= 0:
        return images
    height, width = images[0].shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1.0:
        return images
    size = (round(width * scale), round(height * scale))
    return [cv2.resize(image, size, interpolation=cv2.INTER_AREA) for image in images]


def to_display(image: np.ndarray) -> np.ndarray:
    """Curves a linear image into the 8 bit range that the alignment and the fusion read.

    Both go by how bright a pixel looks rather than how much light reached it, so linear shots are
    encoded first. Shots that arrive encoded already are handed straight back.
    """
    if image.dtype == np.uint8:
        return image
    return (np.clip(image, 0, 1) ** (1 / DISPLAY_GAMMA) * 255).astype(np.uint8)


def align(images: list[np.ndarray]) -> list[np.ndarray]:
    """Shifts the shots onto a common grid, so that a handheld bracket does not come out blurred."""
    # The shift is searched over a pyramid that halves the shots once per bit, and the level it
    # starts from is the one that fixes the result, as the finer levels only refine what that level
    # settled on. Below roughly 32 pixels that level holds too little to match on, so the budget
    # follows the size of the shots, which costs reach on small ones and keeps the shift truthful.
    height, width = images[0].shape[:2]
    max_bits = int(np.clip(np.log2(min(height, width) / 32), 1, 6))
    mtb = cv2.createAlignMTB(max_bits=max_bits)
    # The bitmap a shift is read off is built from 8 bit greys, so the search runs on encoded
    # copies while the shift it settles on is applied to the shots as they actually are. The
    # middle of the bracket is what the rest line up against, being the shot least likely to have
    # lost the detail at either end of the range that the bitmap is cut from.
    greys = [cv2.cvtColor(to_display(image), cv2.COLOR_BGR2GRAY) for image in images]
    middle = greys[len(greys) // 2]
    aligned = []
    for image, grey in zip(images, greys):
        horizontal, vertical = mtb.calculateShift(middle, grey)
        translation = np.float32([[1, 0, horizontal], [0, 1, vertical]])
        aligned.append(cv2.warpAffine(image, translation, (width, height),
                                      borderMode=cv2.BORDER_REPLICATE))
    return aligned
