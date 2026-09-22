"""Command line front end for the SReC codec.

Compress an image and verify that it comes back unchanged::

    python compress.py -c ../experiments/.../model.ckpt -i photo.png

Restore a compressed stream::

    python compress.py -c ../experiments/.../model.ckpt -i photo.srec -o restored.png

Which of the two happens follows from the contents of the input file rather than from its name.

The checkpoint is part of the format: a stream can only be read back with the weights that
wrote it, on a machine that reproduces those predictions bit for bit.
"""

import argparse
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from srec.model.lightning_module import SrecLightningModule
from srec.model.srec_codec import is_stream, load, save


def read_image(path: Path) -> torch.Tensor:
    """Load an image as a 1 x 3 x H x W tensor of integer valued floats in the 0-255 range."""
    pixels = np.array(Image.open(path).convert("RGB"))
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float()


def write_image(image: torch.Tensor, path: Path) -> None:
    to_pil(image).save(path)


def to_pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(image[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy())


def png_size(image: torch.Tensor) -> int:
    """Size in bytes of the same pixels stored as an optimised PNG."""
    buffer = io.BytesIO()
    to_pil(image).save(buffer, format="PNG", optimize=True)
    return buffer.tell()


def load_model(checkpoint: Path) -> torch.nn.Module:
    return SrecLightningModule.load_from_checkpoint(checkpoint, map_location="cpu").srec


def compress(model: torch.nn.Module, image_path: Path, output_path: Path) -> None:
    image = read_image(image_path)
    n_bytes = save(model, image, output_path)

    restored = load(model, output_path)
    if not torch.equal(restored, image):
        raise RuntimeError(f"Round trip of {image_path} is not lossless.")

    reference = png_size(image)
    sub_pixels = image.numel()
    print(f"{image_path} -> {output_path}")
    print(f"  srec: {n_bytes:>8} bytes, {8 * n_bytes / sub_pixels:.4f} bpsp")
    print(f"  png:  {reference:>8} bytes, {8 * reference / sub_pixels:.4f} bpsp")
    print(f"  srec saves {100 * (1 - n_bytes / reference):+.1f}% over png; round trip is lossless")


def decompress(model: torch.nn.Module, stream_path: Path, output_path: Path) -> None:
    write_image(load(model, stream_path), output_path)
    print(f"{stream_path} -> {output_path}")


def main(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        raise SystemExit(f"{args.input} does not exist.")

    restoring = is_stream(args.input)
    output = args.output or args.input.with_suffix(".png" if restoring else ".srec")
    if output == args.input:
        raise SystemExit(f"{output} is both the input and the output; pass --output to write elsewhere.")

    model = load_model(args.checkpoint)
    if restoring:
        decompress(model, args.input, output)
    else:
        compress(model, args.input, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--checkpoint", type=Path, required=True, help="Lightning checkpoint of a trained model.")
    parser.add_argument("-i", "--input", type=Path, required=True,
                        help="Image to compress, or stream to restore; whichever the file turns out to be.")
    parser.add_argument("-o", "--output", type=Path, help="Output path; defaults to the input path with a new suffix.")
    main(parser.parse_args())
