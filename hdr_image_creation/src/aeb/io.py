"""Loading bracket shots and reading exposure metadata."""

from pathlib import Path

import cv2
import numpy as np
import rawpy
from PIL import ExifTags, Image

from aeb.constants import IMAGE_SUFFIXES, RAW_SUFFIXES


def collect_image_paths(inputs: list[Path]) -> list[Path]:
    """Expands directories into the images they hold, leaving the order of explicit paths alone."""
    paths = []
    for entry in inputs:
        if entry.is_dir():
            paths.extend(sorted(p for p in entry.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))
        elif entry.is_file():
            paths.append(entry)
        else:
            raise SystemExit(f"{entry} does not exist.")

    if len(paths) < 2:
        raise SystemExit(f"Combining a bracket takes at least two images, found {len(paths)}.")
    return paths


def load_images(paths: list[Path]) -> list[np.ndarray]:
    """Loads the shots as 8 bit BGR images, which is what the merging operators work on."""
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"{path} could not be read as an image.")
        images.append(image)
    return images


def load_raw_images(paths: list[Path]) -> tuple[list[np.ndarray], np.ndarray]:
    """Develops raw shots into linear images, each paired with the exposure time it was taken at.

    The brightening a converter normally applies per shot is turned off, since it would lift every
    shot to the same level and wipe out the differences the bracket is made of. What comes back
    counts light rather than describing a look, so it stays proportional to the exposure time.
    """
    images, times = [], []
    for path in paths:
        with rawpy.imread(str(path)) as raw:
            developed = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16,
                                        use_camera_wb=True)
            shutter = raw.other.shutter_speed
        if not shutter:
            raise SystemExit(f"{path} records no shutter speed, which leaves fusion mode to combine "
                             "this bracket.")
        images.append(developed[..., ::-1].astype(np.float32) / np.float32(65535))
        times.append(float(shutter))
    return images, np.array(times, dtype=np.float32)


def load_bracket(paths: list[Path]) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Loads the shots, bringing back the exposure times alongside them when they are raw."""
    suffixes = {path.suffix.lower() for path in paths}
    if suffixes & RAW_SUFFIXES:
        if not suffixes <= RAW_SUFFIXES:
            raise SystemExit("The bracket mixes raw shots with encoded ones, which do not combine; "
                             "the two carry light on different scales.")
        images, times = load_raw_images(paths)
    else:
        images, times = load_images(paths), None

    for path, image in zip(paths, images):
        if image.shape != images[0].shape:
            raise SystemExit(f"{path} has shape {image.shape} while {paths[0]} has shape "
                             f"{images[0].shape}; a bracket has to be shot at a single resolution.")
    return images, times


def read_exposure_times(paths: list[Path]) -> np.ndarray:
    """Reads the exposure time of every shot, in seconds, from its EXIF metadata.

    A camera records the time either straight, or as an APEX shutter speed that carries the same
    thing as its negative base two logarithm.
    """
    times = []
    for path in paths:
        with Image.open(path) as image:
            exif = image.getexif().get_ifd(ExifTags.IFD.Exif)
        exposure_time = exif.get(ExifTags.Base.ExposureTime)
        if exposure_time is None:
            shutter_speed = exif.get(ExifTags.Base.ShutterSpeedValue)
            if shutter_speed is None:
                raise SystemExit(f"{path} carries no EXIF exposure time and no shutter speed either, "
                                 "which leaves fusion mode as the way to combine this bracket.")
            exposure_time = 2.0 ** -float(shutter_speed)
        times.append(float(exposure_time))
    return np.array(times, dtype=np.float32)


def validate_exposure_times(times: np.ndarray) -> None:
    if np.any(times <= 0):
        raise SystemExit("Exposure times have to be positive.")
    if len(np.unique(times)) != len(times):
        # A response curve is fitted to how the pixels change between exposures, so repeated
        # exposures carry no information about it.
        raise SystemExit(f"The bracket holds {len(np.unique(times))} distinct exposure times across "
                         f"{len(times)} shots. A camera that brackets on ISO or aperture rather than on "
                         "shutter leaves them equal, which leaves fusion mode to combine this bracket.")
