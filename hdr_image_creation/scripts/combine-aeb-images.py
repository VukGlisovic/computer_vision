"""Combines an auto exposure bracketing (AEB) burst into a single image.

A burst is a set of shots of the same scene, each taken at a different exposure, so that
together they cover a dynamic range that no single shot can hold.

Blend the well exposed regions of every shot straight into a displayable image::

    pixi run python combine-aeb-images.py -i burst_dir/ -o combined.jpg

Recover the camera response curve, merge the shots into a radiance map and tonemap that
map back to a displayable image::

    pixi run python combine-aeb-images.py -i under.jpg normal.jpg over.jpg -m hdr -o combined.jpg

The second route works from the exposure times of the shots, which it takes from their metadata,
and only the ratios between those times reach the result. Shots that carry no usable times are
left to the first route, which asks nothing of the metadata.

Raw shots are developed without the per shot brightening a converter normally applies, which
leaves them counting light rather than describing a look. Their response curve is a straight line
already, so the second route pools their readings directly instead of fitting a curve to undo.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import rawpy
from PIL import ExifTags, Image

# Formats holding pixels a camera has already curved, against formats holding the sensor's own
# readings, which is the difference that decides how a bracket has to be merged.
ENCODED_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
RAW_SUFFIXES = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw"}
IMAGE_SUFFIXES = ENCODED_SUFFIXES | RAW_SUFFIXES

# Readings this far up the range stopped climbing before the shutter closed.
SATURATION = 0.99

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


def to_display(image: np.ndarray) -> np.ndarray:
    """Curves a linear image into the 8 bit range that the alignment and the fusion read.

    Both go by how bright a pixel looks rather than how much light reached it, so linear shots are
    encoded first. Shots that arrive encoded already are handed straight back.
    """
    if image.dtype == np.uint8:
        return image
    return (np.clip(image, 0, 1) ** (1 / DISPLAY_GAMMA) * 255).astype(np.uint8)


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


def merge_linear(images: list[np.ndarray], times: np.ndarray) -> np.ndarray:
    """Merges linear shots into a radiance map by pooling the light each of them collected.

    A reading that never reached the top of the range is light counted over a known interval, so
    the readings add up and the intervals add up with them. Readings that did reach the top stopped
    counting early and are left out, except where a pixel sat at the top throughout, which only the
    shortest shot can say anything about. No response curve enters into it, because a raw shot
    carries no curve to undo.
    """
    signal = np.zeros_like(images[0])
    exposure = np.zeros_like(images[0])
    for image, time in zip(images, times):
        unsaturated = image < SATURATION
        signal += image * unsaturated
        exposure += time * unsaturated

    blown = exposure <= 0
    radiance = np.divide(signal, exposure, out=np.zeros_like(signal), where=~blown)
    shortest = int(np.argmin(times))
    radiance[blown] = (images[shortest] / times[shortest])[blown]
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
    images, times = load_bracket(paths)
    if not args.no_align:
        images = align(images)

    if args.mode == "fusion":
        combined = cv2.createMergeMertens().process([to_display(image) for image in images])
        output = tag_path(args.output, args.mode)
    else:
        if times is None:
            times = read_exposure_times(paths)
            validate_exposure_times(times)
            radiance = merge_hdr(images, times, args.calibration)
            calibration = args.calibration
        else:
            radiance = merge_linear(images, times)
            calibration = "linear"
        if args.radiance:
            # The radiance map is what the shots merge into, which the tonemapping operator only
            # reads afterwards, so the name carries everything up to that point and no further.
            radiance_path = tag_path(args.radiance, args.mode, calibration)
            radiance_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(radiance_path), radiance)
        combined = TONEMAPPERS[args.tonemap](radiance, args.gamma)
        output = tag_path(args.output, args.mode, calibration, args.tonemap)

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
                        help="How to recover the camera response curve. For hdr mode on encoded "
                             "shots; raw shots carry no curve to recover.")
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
