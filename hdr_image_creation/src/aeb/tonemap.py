"""Operators that bring a radiance map back into a displayable range."""

import cv2
import numpy as np

from aeb.constants import (
    ACES_INPUT,
    ACES_OUTPUT,
    BASE_CONTRAST,
    BASE_EDGE,
    LUMINANCE_WEIGHTS,
    MIDDLE_GREY,
)


def aces(radiance: np.ndarray, gamma: float) -> np.ndarray:
    """Tonemaps along the ACES rendering, which rolls the highlights off the way film does.

    The curve is a fixed shape rather than something fitted to the image, and it runs out of room
    a little above mid grey, so the radiance is first keyed to put the scene where the curve
    expects it. The operators that come from the library key themselves and need none of this.
    """
    scene_grey = np.exp(np.mean(np.log(radiance @ LUMINANCE_WEIGHTS + 1e-6)))
    keyed = radiance * (MIDDLE_GREY / scene_grey)
    # The matrices are written for red to blue, which is the reverse of how the images are held.
    channels = keyed[..., ::-1] @ ACES_INPUT.T
    curved = ((channels * (channels + 0.0245786) - 0.000090537)
              / (channels * (0.983729 * channels + 0.432951) + 0.238081))
    return np.clip(curved @ ACES_OUTPUT.T, 0, 1)[..., ::-1] ** (1 / gamma)


def base_layer(log_luminance: np.ndarray) -> np.ndarray:
    """Draws out the broad shape of a scene, keeping the edges that run through it where they are.

    Smoothing that took no account of edges would carry the bright side of one over onto the dark
    side and leave a rim of the wrong brightness along it. So the readings are sorted into slices
    by how bright they are, each slice is smoothed on its own, and every pixel then reads from the
    slices closest to its own reading. What lies across an edge from a pixel falls into slices it
    does not read from, so none of it reaches over.

    A slice holds nothing finer than the shape it was smoothed down to, which a reduced grid has
    room for, while the reading that chooses between the slices is taken at full size. That is
    what holds the edges in place, and it is what makes the whole thing affordable.
    """
    height, width = log_luminance.shape
    reduction = max(1.0, max(height, width) / 512)
    reduced = cv2.resize(log_luminance, (int(width / reduction), int(height / reduction)),
                         interpolation=cv2.INTER_AREA)
    # Around a fiftieth of the frame, which is the scale that parts the shape of a scene from the
    # detail lying on top of it.
    sigma = 0.02 * np.hypot(height, width) / reduction

    slices = np.linspace(log_luminance.min(), log_luminance.max(), 10)
    spacing = max(float(slices[1] - slices[0]), 1e-8)
    gathered = np.zeros_like(log_luminance)
    weight = np.zeros_like(log_luminance)
    for brightness in slices:
        belongs = np.exp(-0.5 * ((reduced - brightness) / BASE_EDGE) ** 2)
        smoothed = (cv2.GaussianBlur(belongs * reduced, (0, 0), sigma)
                    / (cv2.GaussianBlur(belongs, (0, 0), sigma) + 1e-8))
        nearness = np.clip(1 - np.abs(log_luminance - brightness) / spacing, 0, 1)
        gathered += nearness * cv2.resize(smoothed, (width, height),
                                          interpolation=cv2.INTER_LINEAR)
        weight += nearness
    return gathered / np.maximum(weight, 1e-8)


def durand(radiance: np.ndarray, gamma: float) -> np.ndarray:
    """Tonemaps by flattening the broad shape of a scene and leaving the detail on it untouched.

    An operator working from one curve has to spend its range on the distance between the darkest
    and brightest parts of a scene, which leaves little of it over for the texture inside them.
    Parting the two lets that distance be closed on its own, so detail arrives at the strength it
    was captured at wherever in the frame it sits. The cost is that the operator reads the
    surroundings of a pixel rather than the pixel alone, which is what the slices pay for.
    """
    luminance = np.maximum(radiance @ LUMINANCE_WEIGHTS, 1e-6)
    log_luminance = np.log10(luminance)
    base = base_layer(log_luminance)
    detail = log_luminance - base
    compression = np.log10(BASE_CONTRAST) / max(float(base.max() - base.min()), 1e-6)
    # Shifting the flattened base down by its own peak puts the brightest part of the scene at
    # one, which is the top of what a display holds.
    flattened = base * compression + detail - base.max() * compression
    return np.clip(radiance * (10.0 ** flattened / luminance)[..., None], 0, 1) ** (1 / gamma)


# Each operator brings a radiance map back into the nought to one range that a display works in.
TONEMAPPERS = {
    "reinhard": lambda radiance, gamma: cv2.createTonemapReinhard(gamma).process(radiance),
    "mantiuk": lambda radiance, gamma: cv2.createTonemapMantiuk(gamma).process(radiance),
    "drago": lambda radiance, gamma: cv2.createTonemapDrago(gamma).process(radiance),
    "aces": aces,
    "durand": durand,
}
