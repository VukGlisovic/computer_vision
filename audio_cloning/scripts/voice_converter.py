"""
Based on https://github.com/coqui-ai/TTS
"""
import argparse

import torch
from TTS.api import TTS


def main(source_wav: str, target_wav: str, output_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the model
    tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to("cuda")

    # Convert the audio. The voice from the target_wav will be used for the text in the source_wav
    tts.voice_conversion_to_file(source_wav=source_wav, target_wav=target_wav, file_path=output_path)
    print(f"Saved converted file to {output_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_wav", type=str, help="Path to source audio file. The text to use. mp4/ogg are also supported formats.")
    parser.add_argument("-t", "--target_wav", type=str, help="Path to target audio file. The voice to use. mp4/ogg are also supported formats.")
    parser.add_argument("-o", "--output_path", type=str, help="Where to store the converted audio file (wav/mp4 format).")
    args = parser.parse_args()

    main(args.source_wav, args.target_wav, args.output_path)
