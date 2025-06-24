import os
import re
import argparse
from pathlib import Path

import imageio


def natural_sort_key(s: str) -> list[str | int]:
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def create_gif_from_folder(folder_path: Path, output_path: Path, frame_duration: int = 200) -> None:
    # Get all PNG files and sort them naturally
    img_files = [f for f in os.listdir(folder_path)]
    img_files.sort(key=natural_sort_key)
    
    # Create list of full file paths
    images = []
    for filename in img_files:
        file_path = os.path.join(folder_path, filename)
        images.append(imageio.imread(file_path))
    
    # Save as GIF (automatically done based on output file extension)
    imageio.mimsave(output_path, images, duration=frame_duration)


def main():
    parser = argparse.ArgumentParser(description='Create GIF from generated samples')
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                      help='Directory containing the generated samples')
    parser.add_argument('-o', '--output_name', type=str, default='training_progress.gif',
                      help='Name of the output GIF file (default: training_progress.gif)')
    parser.add_argument('-d', '--frame_duration', type=int, default=200,
                      help='Duration of each frame in milliseconds (default: 200)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise ValueError(f"Input directory {input_dir} does not exist")
    
    output_path = input_dir.parent / args.output_name
    print(f"Creating GIF from {input_dir}...")
    create_gif_from_folder(input_dir, output_path, args.frame_duration)
    print(f"GIF saved to {output_path}")


if __name__ == "__main__":
    main()
