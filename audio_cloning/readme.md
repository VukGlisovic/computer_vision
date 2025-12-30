# Audio cloning

My main goal here was to be able to clone voices sothat I can make a funny pubquiz by mixing voices of friends.
There's currently two ways of doing this in this repo:
1. Use `voice_converter.py`. In this script, a source and target audio file is expected as input. The voice of the 
   target audio will be used to "cover" the audio of the source audio. Basically you will have the source audio with all
   its text and pronunciations, but with the voice of the target audio.
2. Use `speech_to_text.py` in combination with `text_to_speech.py`. In this scenario, you first convert the audio of the
   source to text. Then you input that text together with a target audio (containing the voice you want). Now you can 
   basically make the target voice say anything you want (but in particular what was said in the source audio).
