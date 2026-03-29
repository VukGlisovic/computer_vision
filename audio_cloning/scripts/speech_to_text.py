"""
Based on https://github.com/openai/whisper
"""
import argparse

import whisper


def main(input_audio: str, output_path: str):
    # Load the model
    model = whisper.load_model("turbo")

    # Transcribe audio to text
    result = model.transcribe(input_audio)

    # Print and store the transcription
    transcription = result['text']
    print(transcription)
    with open(output_path, "w") as f:
        f.write(transcription)
    print(f"Transcription written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_audio", type=str, help="Path to input audio file. The text to use. mp4/ogg are also supported formats.")
    parser.add_argument("-o", "--output_path", type=str, help="Where to store the transcribed audio (.txt file).")
    args = parser.parse_args()

    main(args.input_audio, args.output_path)
