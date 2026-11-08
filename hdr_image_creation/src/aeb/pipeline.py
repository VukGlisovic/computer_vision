"""Orchestrates loading, merging and writing a combined bracket image."""

from pathlib import Path

import cv2
import numpy as np

from aeb.io import (
    collect_image_paths,
    load_bracket,
    read_exposure_times,
    validate_exposure_times,
)
from aeb.merge import merge_hdr, merge_linear
from aeb.preprocess import align, downscale, to_display
from aeb.tonemap import TONEMAPPERS


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Maps a float image in the 0-1 range onto the 0-255 range, clipping whatever falls outside.

    Tonemapping operators leave a NaN behind here and there, which has no meaning as a pixel value
    and turns into an arbitrary one when cast, so it is pinned to black.
    """
    return np.clip(np.nan_to_num(image * 255), 0, 255).astype(np.uint8)


def tag_path(path: Path, *settings: str) -> Path:
    """Works the settings that shaped a file into its name, so that runs do not overwrite each other."""
    return path.with_stem("_".join([path.stem, *settings]))


def combine(
    inputs: list[Path],
    output: Path,
    mode: str = "fusion",
    calibration: str = "debevec",
    tonemap: str = "reinhard",
    gamma: float = 2.2,
    radiance: Path | None = None,
    no_align: bool = False,
    max_side: int | None = None,
) -> tuple[Path, int]:
    """Combines a bracket into a displayable image.

    Returns the path that was written and how many shots went into it.
    """
    paths = collect_image_paths(inputs)
    images, times = load_bracket(paths)
    images = downscale(images, max_side)
    if not no_align:
        images = align(images)

    if mode == "fusion":
        combined = cv2.createMergeMertens().process([to_display(image) for image in images])
        written = tag_path(output, mode)
    else:
        if times is None:
            times = read_exposure_times(paths)
            validate_exposure_times(times)
            radiance_map = merge_hdr(images, times, calibration)
            used_calibration = calibration
        else:
            radiance_map = merge_linear(images, times)
            used_calibration = "linear"
        if radiance:
            # The radiance map is what the shots merge into, which the tonemapping operator only
            # reads afterwards, so the name carries everything up to that point and no further.
            radiance_path = tag_path(radiance, mode, used_calibration)
            radiance_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(radiance_path), radiance_map)
        combined = TONEMAPPERS[tonemap](radiance_map, gamma)
        written = tag_path(output, mode, used_calibration, tonemap)

    written.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(written), to_uint8(combined)):
        raise SystemExit(f"{written} could not be written.")
    return written, len(paths)
