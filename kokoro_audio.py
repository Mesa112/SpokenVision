from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf


def load_kokoro_model(lang_code='a'):
    return KPipeline(lang_code=lang_code)



def text_to_audio(model, input_text, output_dir='./audio_output', file_name = "audio_output", voice='af_heart'):
    generator = model(input_text, voice=voice)
    audio_files = []

    for i, (gs, ps, audio) in enumerate(generator):
        print(i, gs, ps)
        audio_file = f'{output_dir}/{file_name}.wav'
        sf.write(audio_file, audio, 24000)
        audio_files.append(audio_file)

    return audio_files