"""Combines an auto exposure bracketing (AEB) burst into a single image.

A burst is a set of shots of the same scene, each taken at a different exposure, so that
together they cover a dynamic range that no single shot can hold.

Blend the well exposed regions of every shot straight into a displayable image::

    pixi run python combine-aeb-images.py -i burst_dir/ -o combined.jpg

Recover the camera response curve, merge the shots into a radiance map and tonemap that
map back to a displayable image::

    pixi run python combine-aeb-images.py -i under.jpg normal.jpg over.jpg -m hdr -o combined.jpg

The second route works from the exposure times of the shots, which it takes from their EXIF
metadata, and only the ratios between those times reach the result. Shots that carry no usable
times are left to the first route, which asks nothing of the metadata.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

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


# Each operator brings a radiance map back into the nought to one range that a display works in.
TONEMAPPERS = {
    "reinhard": lambda radiance, gamma: cv2.createTonemapReinhard(gamma).process(radiance),
    "mantiuk": lambda radiance, gamma: cv2.createTonemapMantiuk(gamma).process(radiance),
    "drago": lambda radiance, gamma: cv2.createTonemapDrago(gamma).process(radiance),
    "aces": aces,
}


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
        if images and image.shape != images[0].shape:
            raise SystemExit(f"{path} has shape {image.shape} while {paths[0]} has shape {images[0].shape}; "
                             "a bracket has to be shot at a single resolution.")
        images.append(image)
    return images


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


def align(images: list[np.ndarray]) -> list[np.ndarray]:
    """Shifts the shots onto a common grid, so that a handheld bracket does not come out blurred."""
    # The shift is searched over a pyramid that halves the shots once per bit, and the level it
    # starts from is the one that fixes the result, as the finer levels only refine what that level
    # settled on. Below roughly 32 pixels that level holds too little to match on, so the budget
    # follows the size of the shots, which costs reach on small ones and keeps the shift truthful.
    height, width = images[0].shape[:2]
    max_bits = int(np.clip(np.log2(min(height, width) / 32), 1, 6))
    # The destination has to come in at full size, as the operator writes into it rather than
    # growing it, and it has to own its pixels, as the shots are read while it is written.
    aligned = [image.copy() for image in images]
    cv2.createAlignMTB(max_bits=max_bits).process(images, aligned)
    return aligned


def merge_hdr(images: list[np.ndarray], times: np.ndarray, calibration: str) -> np.ndarray:
    """Merges the shots into a radiance map holding the scene brightness as a float per sub-pixel."""
    response = CALIBRATORS[calibration]().process(images, times)
    radiance = MERGERS[calibration]().process(images, times, response)
    if not np.isfinite(radiance).all():
        # A fit that diverges fills the curve with NaN, which spreads over the whole radiance map
        # and leaves the tonemapping operators to fail on it deep down.
        raise SystemExit(f"The {calibration} response curve did not converge on this bracket. "
                         "A bracket that covers more of the tonal range, or the other calibration, may fit.")
    return radiance


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Maps a float image in the 0-1 range onto the 0-255 range, clipping whatever falls outside.

    Tonemapping operators leave a NaN behind here and there, which has no meaning as a pixel value
    and turns into an arbitrary one when cast, so it is pinned to black.
    """
    return np.clip(np.nan_to_num(image * 255), 0, 255).astype(np.uint8)


def tag_path(path: Path, *settings: str) -> Path:
    """Works the settings that shaped a file into its name, so that runs do not overwrite each other."""
    return path.with_stem("_".join([path.stem, *settings]))


def main(args: argparse.Namespace) -> None:
    paths = collect_image_paths(args.input)
    images = load_images(paths)
    if not args.no_align:
        images = align(images)

    if args.mode == "fusion":
        combined = cv2.createMergeMertens().process(images)
        output = tag_path(args.output, args.mode)
    else:
        times = read_exposure_times(paths)
        validate_exposure_times(times)
        radiance = merge_hdr(images, times, args.calibration)
        if args.radiance:
            # The radiance map is what the shots merge into, which the tonemapping operator only
            # reads afterwards, so the name carries everything up to that point and no further.
            radiance_path = tag_path(args.radiance, args.mode, args.calibration)
            radiance_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(radiance_path), radiance)
        combined = TONEMAPPERS[args.tonemap](radiance, args.gamma)
        output = tag_path(args.output, args.mode, args.calibration, args.tonemap)

    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), to_uint8(combined)):
        raise SystemExit(f"{output} could not be written.")
    print(f"Combined {len(paths)} images into {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", type=Path, nargs="+", required=True,
                        help="Images of the bracket, and/or directories holding them.")
    parser.add_argument("-o", "--output", type=Path, default=Path("combined.jpg"),
                        help="Where to write the combined image, under a name that picks up the settings "
                             "it was made with.")
    parser.add_argument("-m", "--mode", choices=("fusion", "hdr"), default="fusion",
                        help="Blend the shots directly, or merge them into a radiance map and tonemap that.")
    parser.add_argument("-c", "--calibration", choices=tuple(CALIBRATORS), default="debevec",
                        help="How to recover the camera response curve. For hdr mode.")
    parser.add_argument("-t", "--tonemap", choices=tuple(TONEMAPPERS), default="reinhard",
                        help="Operator that brings the radiance map back into a displayable range. For hdr mode.")
    parser.add_argument("-g", "--gamma", type=float, default=2.2,
                        help="Gamma applied by the tonemapping operator. For hdr mode.")
    parser.add_argument("-r", "--radiance", type=Path,
                        help="Where to additionally write the radiance map, as .hdr, under a name that picks "
                             "up the settings it was made with. For hdr mode.")
    parser.add_argument("--no-align", action="store_true",
                        help="Keep the shots as they are instead of aligning them onto a common grid.")
    main(parser.parse_args())
