"""
Main entry point for the Oi-Chatbot Speech-to-Text system.

- Starts background audio, transcript writer, and performance monitor threads
- Launches the FastAPI server for API access
- NOTE: This folder was renamed from 'Components' to 'Speech_to_text'.
"""

import threading
import uvicorn
from audio import select_input_device, audio_thread
from models import get_models
from speaker import identify_speaker, load_known_speakers
from transcript import log_transcript, periodic_writer
from utils import performance_monitor
from api import app
import queue

# === CONFIG ===
samplerate = 16000
blocksize = 16000
speaker_threshold = 0.55

# === MAIN ENTRY POINT ===
def process_chunk(audio_frames):
    import numpy as np
    import io
    from scipy.io.wavfile import write
    import time
    from utils import latency_data
    encoder, whisper_model, _ = get_models()
    known_speakers = load_known_speakers()
    start = time.time()
    print("[🧠 Processing audio chunk...]")
    full_chunk = np.concatenate(audio_frames)
    if np.mean(np.abs(full_chunk)) < 0.01:
        print("[🔇 Silence skipped]")
        return
    wav_io = io.BytesIO()
    write(wav_io, samplerate, (full_chunk * 32767).astype(np.int16))
    wav_io.seek(0)
    try:
        segments, _ = whisper_model.transcribe(
            wav_io,
            vad_filter=True,
            vad_parameters={"threshold": 0.6, "min_silence_duration_ms": 300}
        )
        print("[🔍 Raw Whisper Output]:", segments)
        for segment in segments:
            text = segment.text.strip()
            if text:
                print(f"[📄 Transcribed] {text}")
                speaker = identify_speaker(audio_frames, encoder, known_speakers, speaker_threshold, samplerate)
                log_transcript(speaker, text)
    except Exception as e:
        print(f"[❌ Whisper Error] {e}")
    end = time.time()
    latency_data.append(end - start)

def start_background_tasks():
    q = queue.Queue()
    device = select_input_device()
    print(f"🎧 Using device: {device}")
    print("🚀 FastAPI server running at http://localhost:9575")
    threading.Thread(target=audio_thread, args=(q, process_chunk, samplerate, blocksize, device), daemon=True).start()
    threading.Thread(target=periodic_writer, daemon=True).start()
    threading.Thread(target=performance_monitor, daemon=True).start()

if __name__ == "__main__":
    start_background_tasks()
    uvicorn.run(app, host="0.0.0.0", port=9575, log_level="info")
