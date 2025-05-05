from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import numpy as np


def load_kokoro_model(lang_code='a'):
    return KPipeline(lang_code=lang_code)



def text_to_audio(model, input_text, output_dir='./audio_output', file_name = "audio_output", voice='af_heart'):
    generator = model(input_text, voice=voice)
    all_audio = []

    for i, (gs, ps, audio) in enumerate(generator):
        print("\ngenerated audio ", i, gs)
        all_audio.append(audio) 

    # Concatenate all audio segments into one array
    combined_audio = np.concatenate(all_audio, axis=0)

    # Write the combined audio to a single file
    audio_file = f'{output_dir}/{file_name}.wav'
    sf.write(audio_file, combined_audio, 24000)

    return audio_file