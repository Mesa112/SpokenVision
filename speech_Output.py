import os
import sys
import numpy as np
from datetime import datetime
from gtts import gTTS  # Google Text-to-Speech
import pygame  # For audio playback
import tempfile
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def enhance_description_with_gpt(raw_description):
    """Use GPT-3.5-Turbo to convert technical descriptions into natural, conversational speech."""
    if not os.getenv("OPENAI_API_KEY"):
        return raw_description
    
    try:
        # Create a system prompt for natural speech conversion
        system_prompt = """You are an assistive technology that describes what a visually impaired person is seeing through their camera. 
Convert the technical scene description into natural, helpful speech that would assist someone who cannot see.
Focus on:
- Using conversational tone
- Describing important objects and their spatial relationships naturally
- Providing useful context that helps the person navigate and understand their environment
- Being concise but informative
- Using directional terms (left, right, in front, behind) that would be helpful for orientation
- Mentioning any potential hazards or important changes in the scene"""

        # Call GPT-3.5-Turbo to enhance the description
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using GPT-3.5-Turbo for cost efficiency
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Convert this technical description into natural speech: {raw_description}"}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        enhanced_description = response.choices[0].message.content.strip()
        return enhanced_description
    
    except Exception as e:
        print(f"Error enhancing description with GPT: {e}")
        return raw_description

class SpeechOutput:
    """
    An enhanced speech output module supporting multiple TTS backends including OpenAI.
    """
    
    def __init__(self, output_dir='./audio_output', use_playback=True):
        """
        Initialize the speech output module.
        
        Args:
            output_dir: Directory to save audio files
            use_playback: Whether to play audio immediately after generation
        """
        self.output_dir = output_dir
        self.use_playback = use_playback
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Initialize pygame mixer for audio playback if needed
        if self.use_playback:
            try:
                pygame.mixer.init()
            except:
                print("Warning: Could not initialize pygame mixer for audio playback")
                self.use_playback = False
    
    def text_to_speech_openai(self, text, file_name=None, voice='coral', model="gpt-4o-mini-tts", instructions=None):
        """
        Convert text to speech using OpenAI's TTS API.
        Note: This TTS API might not be available for all accounts.
        """
        if file_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"speech_{timestamp}"
        
        file_path = Path(self.output_dir) / f"{file_name}.mp3"
        
        try:
            # Limit text length to avoid API issues
            if len(text) > 4000:
                print(f"Warning: Text too long ({len(text)} chars), truncating to 4000 chars")
                text = text[:4000]
            
            # Prepare the request parameters
            params = {
                "model": model,
                "voice": voice,
                "input": text
            }
            
            # Add instructions if provided and model supports it
            if instructions and model == "gpt-4o-mini-tts":
                params["instructions"] = instructions
            
            # Generate speech using OpenAI with streaming response
            with openai_client.audio.speech.with_streaming_response.create(**params) as response:
                response.stream_to_file(str(file_path))
            
            print(f"Audio saved to {file_path}")
            
            # Play the audio if playback is enabled
            if self.use_playback:
                self._play_audio(str(file_path))
            
            return str(file_path)
        
        except Exception as e:
            print(f"Error generating OpenAI speech: {e}")
            return None
    
    def text_to_speech_gtts(self, text, file_name=None, lang='en', enhance_text=True):
        """
        Convert text to speech using Google Text-to-Speech.
        
        Args:
            text: Text to convert to speech
            file_name: Output file name (without extension)
            lang: Language code (default: 'en' for English)
            enhance_text: Whether to enhance the text with GPT before TTS
            
        Returns:
            Path to the generated audio file
        """
        if file_name is None:
            # Generate timestamp-based filename if none provided
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"speech_{timestamp}"
        
        file_path = os.path.join(self.output_dir, f"{file_name}.mp3")
        
        try:
            # Enhance text with GPT-3.5-Turbo if requested
            if enhance_text:
                original_text = text
                enhanced_text = enhance_description_with_gpt(text)
                if enhanced_text != original_text:
                    print(f"Text enhanced by GPT-3.5-Turbo")
                    text = enhanced_text
            
            # Limit text length to avoid API issues
            if len(text) > 5000:
                print(f"Warning: Text too long ({len(text)} chars), truncating to 5000 chars")
                text = text[:5000]
            
            # Generate speech using gTTS
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(file_path)
            
            print(f"Audio saved to {file_path}")
            
            # Play the audio if playback is enabled
            if self.use_playback:
                self._play_audio(file_path)
            
            return file_path
        
        except Exception as e:
            print(f"Error generating speech: {e}")
            return None
    
    def text_to_speech_pyttsx3(self, text, file_name=None, enhance_text=True):
        """
        Convert text to speech using pyttsx3 (offline TTS).
        This is a backup option that works without internet.
        
        Args:
            text: Text to convert to speech
            file_name: Output file name (without extension)
            enhance_text: Whether to enhance the text with GPT before TTS
            
        Returns:
            Path to the generated audio file
        """
        try:
            import pyttsx3
        except ImportError:
            print("pyttsx3 not installed. Run 'pip install pyttsx3' to use offline TTS")
            return None
            
        if file_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"speech_{timestamp}"
        
        file_path = os.path.join(self.output_dir, f"{file_name}.mp3")
        
        try:
            # Enhance text with GPT-3.5-Turbo if requested
            if enhance_text:
                original_text = text
                enhanced_text = enhance_description_with_gpt(text)
                if enhanced_text != original_text:
                    print(f"Text enhanced by GPT-3.5-Turbo")
                    text = enhanced_text
            
            engine = pyttsx3.init()
            engine.save_to_file(text, file_path)
            engine.runAndWait()
            
            print(f"Audio saved to {file_path}")
            
            # Play the audio if playback is enabled
            if self.use_playback:
                self._play_audio(file_path)
            
            return file_path
        
        except Exception as e:
            print(f"Error generating speech with pyttsx3: {e}")
            return None
    
    def _play_audio(self, audio_path):
        """
        Play audio file using pygame mixer.
        
        Args:
            audio_path: Path to audio file
        """
        try:
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            print(f"Error playing audio: {e}")


# Updated compatibility function with OpenAI option
def text_to_audio(model, input_text, output_dir='./audio_output', file_name="audio_output", 
                 voice='coral', tts_backend='openai', instructions=None, tts_model="gpt-4o-mini-tts",
                 enhance_text=True):
    """
    Enhanced compatibility function that supports multiple TTS backends.
    
    Args:
        model: Not used in this implementation (kept for compatibility)
        input_text: Text to convert to speech
        output_dir: Directory to save audio files
        file_name: Output file name
        voice: Voice ID for OpenAI (11 options) or language code for gTTS
        tts_backend: 'openai', 'gtts', or 'pyttsx3'
        instructions: Optional instructions for OpenAI TTS (only works with gpt-4o-mini-tts)
        tts_model: OpenAI TTS model to use (gpt-4o-mini-tts, tts-1, or tts-1-hd)
        enhance_text: Whether to enhance the text with GPT-3.5-Turbo before TTS
        
    Returns:
        List of generated audio files
    """
    speech_output = SpeechOutput(output_dir=output_dir)
    
    if tts_backend == 'openai':
        # For OpenAI, ensure voice is one of the valid options
        valid_voices = ['alloy', 'ash', 'ballad', 'coral', 'echo', 'fable', 
                       'nova', 'onyx', 'sage', 'shimmer']
        if voice not in valid_voices:
            print(f"Warning: Invalid voice '{voice}'. Using 'coral' instead.")
            voice = 'coral'
        
        # For OpenAI TTS, we'll enhance first then use TTS
        text_to_speak = input_text
        if enhance_text:
            text_to_speak = enhance_description_with_gpt(input_text)
        
        audio_file = speech_output.text_to_speech_openai(
            text_to_speak, file_name, voice=voice, model=tts_model, instructions=instructions
        )
    elif tts_backend == 'gtts':
        # For gTTS, voice parameter is used as language code
        lang = 'en'  # Default to English
        if voice.startswith('fr'):
            lang = 'fr'
        elif voice.startswith('es'):
            lang = 'es'
        elif voice.startswith('de'):
            lang = 'de'
        audio_file = speech_output.text_to_speech_gtts(input_text, file_name, lang=lang, enhance_text=enhance_text)
    elif tts_backend == 'pyttsx3':
        audio_file = speech_output.text_to_speech_pyttsx3(input_text, file_name, enhance_text=enhance_text)
    else:
        print(f"Unknown TTS backend: {tts_backend}. Using GTTs by default.")
        audio_file = speech_output.text_to_speech_gtts(input_text, file_name, lang='en', enhance_text=enhance_text)
    
    return [audio_file] if audio_file else []


# Simple placeholder function to match the original Kokoro interface
def load_kokoro_model(lang_code='a'):
    """
    Placeholder function to match the original Kokoro interface.
    Returns a dummy model object that the text_to_audio function can use.
    """
    return {"lang": lang_code}  # Return a simple dict as a mock model


# Demo/test function
def test_speech_output():
    """Test the speech output module with different backends."""
    speech = SpeechOutput()
    
    # Test with GPT-3.5-Turbo enhancement + gTTS (most cost-effective)
    speech.text_to_speech_gtts(
        "This is a technical test. Object detected: person at center position, medium distance. "
        "New objects: chair at bottom left, very close.",
        "test_enhanced_gtts",
        enhance_text=True
    )
    
    # Test with enhancement disabled
    speech.text_to_speech_gtts(
        "This is a technical test. Object detected: person at center position, medium distance. "
        "New objects: chair at bottom left, very close.",
        "test_raw_gtts",
        enhance_text=False
    )
    
    # Test with OpenAI TTS (if available)
    try:
        speech.text_to_speech_openai(
            "This is a test of the OpenAI Text-to-Speech integration.",
            "test_openai_fallback",
            voice="coral",
            model="tts-1",
            instructions="Speak in a cheerful tone."
        )
    except:
        print("OpenAI TTS test skipped - not available for this account")
    
    # Test with pyttsx3 (if available)
    try:
        speech.text_to_speech_pyttsx3(
            "Object detected: person in front, 5 meters away.",
            "test_enhanced_pyttsx3",
            enhance_text=True
        )
    except:
        print("pyttsx3 test skipped - library not available")
    
    # Test the compatibility function with GPT enhancement
    model = load_kokoro_model()
    text_to_audio(
        model,
        "Technical description: Person detected at position 320,240. Distance 3 meters.",
        file_name="test_compat_enhanced",
        tts_backend="gtts",
        voice="en",
        enhance_text=True
    )


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Note: OPENAI_API_KEY not set. GPT enhancement will not work.")
    test_speech_output()