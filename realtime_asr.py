import deepspeech
import numpy as np
import pyaudio

def create_model(model_path, scorer_path):
    model = deepspeech.Model(model_path)
    model.enableExternalScorer(scorer_path)
    return model

def create_audio_stream(rate, channels):
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=1024
    )
    return stream

def main():
    model_path = "F:\DeepSpeech_project\deepspeech-0.9.3-models.pbmm"
    scorer_path = "F:\DeepSpeech_project\deepspeech-0.9.3-models.scorer"
    rate = 16000
    channels = 1

    model = create_model(model_path, scorer_path)
    stream = create_audio_stream(rate, channels)

    print("Listening...")

    # Stream audio in chunks for real-time processing
    while True:
        audio_chunk = np.frombuffer(stream.read(1024), dtype=np.int16)
        if len(audio_chunk) == 0:
            continue

        # Perform inference on the audio chunk
        text = model.stt(audio_chunk)

        # Print the transcribed text in real-time
        print("Transcription:", text)

    # Close the stream and release resources
    stream.stop_stream()
    stream.close()

if __name__ == "__main__":
    main()
