import deepspeech
import numpy as np
from pydub import AudioSegment

def create_model(model_path, scorer_path):
    model = deepspeech.Model(model_path)
    model.enableExternalScorer(scorer_path)
    return model

def transcribe_audio(model, audio_chunk):
    return model.stt(audio_chunk)

def read_wav_file(file_path):
    audio = AudioSegment.from_file(file_path)
    return np.array(audio.get_array_of_samples(), dtype=np.int16)

def main():
    model_path = r"F:\DeepSpeech_project\deepspeech-0.9.3-models.pbmm"
    scorer_path = r"F:\DeepSpeech_project\deepspeech-0.9.3-models.scorer"
    file_path = r"F:\DeepSpeech_project\Power_English_Update.mp3"

    model = create_model(model_path, scorer_path)

    # Read audio from a file
    audio_chunk = np.frombuffer(read_wav_file(file_path), dtype=np.int16)

    # Perform inference on the audio chunk
    transcription = transcribe_audio(model, audio_chunk)

    # Print the transcribed text
    print("Transcription:", transcription)

if __name__ == "__main__":
    main()
