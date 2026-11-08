"""Command-line entry for combining an AEB bracket."""

import argparse
from pathlib import Path

from aeb.constants import CALIBRATORS
from aeb.pipeline import combine
from aeb.tonemap import TONEMAPPERS

DOC = """Combines an auto exposure bracketing (AEB) burst into a single image.

A burst is a set of shots of the same scene, each taken at a different exposure, so that
together they cover a dynamic range that no single shot can hold.

Blend the well exposed regions of every shot straight into a displayable image:

    pixi run combine-aeb-images -i burst_dir/ -o combined.jpg

Recover the camera response curve, merge the shots into a radiance map and tonemap that
map back to a displayable image:

    pixi run combine-aeb-images -i under.jpg normal.jpg over.jpg -m hdr -o combined.jpg

The second route works from the exposure times of the shots, which it takes from their metadata,
and only the ratios between those times reach the result. Shots that carry no usable times are
left to the first route, which asks nothing of the metadata.

Raw shots are developed without the per shot brightening a converter normally applies, which
leaves them counting light rather than describing a look. Their response curve is a straight line
already, so the second route pools their readings directly instead of fitting a curve to undo.
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    parser.add_argument("--max-side", type=int,
                        help="Downscale the shots so their longer side stays under this many pixels, "
                             "which speeds a run up by giving it less to work over. Left unset, or set to "
                             "zero, the shots keep their full size.")
    args = parser.parse_args(argv)
    written, count = combine(
        inputs=args.input,
        output=args.output,
        mode=args.mode,
        calibration=args.calibration,
        tonemap=args.tonemap,
        gamma=args.gamma,
        radiance=args.radiance,
        no_align=args.no_align,
        max_side=args.max_side,
    )
    print(f"Combined {count} images into {written}")


if __name__ == "__main__":
    main()
