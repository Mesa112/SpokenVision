from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf


def text_to_audio(input_text, lang_code='a', voice='af_heart', output_dir='.'):
    """
    Converts input text to audio using the KPipeline.

    Args:
        input_text (str): The text to convert to audio.
        lang_code (str): Language code for the pipeline. Default is 'a'.
        voice (str): Voice to use for audio generation. Default is 'af_heart'.
        output_dir (str): Directory to save the audio files. Default is the current directory.

    Returns:
        list: A list of file paths to the generated audio files.
    """
    pipeline = KPipeline(lang_code=lang_code)
    generator = pipeline(input_text, voice=voice)
    audio_files = []

    for i, (gs, ps, audio) in enumerate(generator):
        print(i, gs, ps)
        audio_file = f'{output_dir}/{i}.wav'
        sf.write(audio_file, audio, 24000)
        audio_files.append(audio_file)

    return audio_files