"""Merging a bracket into a radiance map."""

import numpy as np

from aeb.constants import CALIBRATORS, LEVELS, MERGERS, SATURATION, SATURATED


def merge_hdr(images: list[np.ndarray], times: np.ndarray, calibration: str) -> np.ndarray:
    """Merges the shots into a radiance map holding the scene brightness as a float per sub-pixel.

    A merge trusts a reading by how far it sits from either end of the range, so where every shot
    in the bracket reached the top there is nothing left to trust, and what the merge settles on
    there is drawn from the long shots as much as the short ones. That puts the sun below the sky
    beside it, which is still holding readings the shortest shot caught in full. So the pixels
    that reached the top throughout are set apart from the merge and given the most that the
    bracket can account for.
    """
    response = CALIBRATORS[calibration]().process(images, times)
    radiance = MERGERS[calibration]().process(images, times, response)
    if not np.isfinite(radiance).all():
        # A fit that diverges fills the curve with NaN, which spreads over the whole radiance map
        # and leaves the tonemapping operators to fail on it deep down.
        raise SystemExit(f"The {calibration} response curve did not converge on this bracket. "
                         "A bracket that covers more of the tonal range, or the other calibration, may fit.")

    blown = np.ones(radiance.shape, dtype=bool)
    for image in images:
        blown &= image >= SATURATED
    if not blown.any() or blown.all():
        return radiance

    # The top of the curve over the shortest shot is the least light a blown pixel can have
    # caught. A curve fitted where no shot held detail can come out lower there than further
    # down, so the brightest pixel the bracket did account for sits underneath it as a floor.
    from_curve = response.reshape(LEVELS, -1)[SATURATED] / float(times.min())
    ceiling = np.maximum(from_curve, radiance[~blown].max()).astype(np.float32)
    return np.where(blown, ceiling, radiance)


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
