"""
Based on https://github.com/resemble-ai/chatterbox

multilingual_model.generate has a hardcoded limit of 1000 output tokens. If you need
more, you can change this in audio_cloning/.pixi/envs/chatterbox/lib/python3.11/site-packages/chatterbox/mtl_tts.py
"""
import os
import argparse

import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


def main(text_input: str, audio_prompt_path: str, output_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load text to create audio for
    if os.path.exists(text_input):
        print(f"Loading text from input file: {text_input}")
        with open(text_input, "r") as f:
            text = f.read()
    else:
        print("text_input is not a file, assuming it's the text itself.")
        text = text_input

    # Load the multilingual model (this one also supports Dutch)
    multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    # Generate audio (requires a reference clip for voice cloning)
    wav = multilingual_model.generate(text, audio_prompt_path=audio_prompt_path, language_id='nl')

    # Save generated audio to disk
    # multilingual_model.sr = sampling rate. It basically controls the speed at which the audio will be played. If
    # you increase the sampling rate compared to its original, then the pitch of voice will sound higher. If you
    # decrease sampling rate, the pitch will sound lower.
    ta.save(output_path, wav, multilingual_model.sr)
    print(f"Wrote converted text to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--text_input", type=str, help="Path to the text input file or text directly (distinguished based on .txt extension).")
    parser.add_argument("-a", "--audio_prompt_path", type=str, help="Path to target audio file. The voice to use. mp4/ogg are also supported formats.")
    parser.add_argument("-o", "--output_path", type=str, help="Where to store the converted audio file (wav/mp4 format).")
    args = parser.parse_args()

    main(args.text_input, args.audio_prompt_path, args.output_path)
