import queue
import sys
import os
import threading
import numpy as np
import sounddevice as sd
import torch
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
from collections import deque
from sklearn.cluster import AgglomerativeClustering
from faster_whisper import WhisperModel
from scipy.io.wavfile import write
import io
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import uvicorn
import psutil

# === CONFIG ===
transcript_dir = "E:/Hackathon/Transcripts"
os.makedirs(transcript_dir, exist_ok=True)
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
speaker_transcript_path = os.path.join(transcript_dir, f"with_speakers_{timestamp_str}.txt")
plain_transcript_path = os.path.join(transcript_dir, f"plain_{timestamp_str}.txt")

# === AUTO-SELECT MIC DEVICE ===
device = None
try:
    devices = sd.query_devices()
    for idx, d in enumerate(devices):
        if d['max_input_channels'] >= 1:
            print(f"[🎤 Found input device] ID {idx}: {d['name']}")
            device = idx
            break
    if device is None:
        raise RuntimeError("No suitable input device with 1+ channels.")
except Exception as e:
    print(f"[❌ Audio Device Error] {e}")
    device = None

samplerate = 16000
blocksize = 16000
speaker_threshold = 0.55
MAX_TRANSCRIPT_LINES = 10000

# === MODELS ===
device_type = "cuda" if torch.cuda.is_available() else "cpu"
encoder = VoiceEncoder().to(device_type)
whisper_model = WhisperModel("medium", device=device_type, compute_type="float16" if device_type == "cuda" else "int8")

# === STORAGE ===
known_speakers = {}
try:
    if os.path.exists("known_speakers.pkl"):
        with open("known_speakers.pkl", "rb") as f:
            known_speakers = pickle.load(f)
except Exception as e:
    print(f"[⚠️ Failed to load known_speakers.pkl] {e}. Regenerating...")
    known_speakers = {}

recent_predictions = []
q = queue.Queue()
transcript_lines_speaker = deque(maxlen=MAX_TRANSCRIPT_LINES)
transcript_lines_plain = deque(maxlen=MAX_TRANSCRIPT_LINES)
transcript_lock = threading.Lock()
write_buffer = []
executor = ThreadPoolExecutor(max_workers=2)
latency_data = deque(maxlen=100)

# === AUDIO ===
def callback(indata, frames, time, status):
    if status:
        print("[⚠️ Audio Status]", status, file=sys.stderr)
    q.put(bytes(indata))


# Store past embeddings to improve clustering accuracy
speaker_embedding_history = deque(maxlen=20)  # Keeps last 20 embeddings

def cluster_speakers():
    """Clusters speaker embeddings to dynamically differentiate voices."""
    if len(speaker_embedding_history) < 5:  # Need enough samples for clustering
        return None

    embeddings = np.array(speaker_embedding_history)
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.4, linkage="ward")
    labels = clustering_model.fit_predict(embeddings)

    return labels[-1]  # Return most recent cluster assignment

def identify_speaker(audio_frames, encoder, known_speakers, speaker_threshold, samplerate):
    """Identifies speakers using adaptive learning and spectral clustering."""
    
    wav = preprocess_wav(np.concatenate(audio_frames), source_sr=samplerate)

    if np.mean(np.abs(wav)) < 0.01:  # Ignore silence
        return "Unknown"

    embedding = encoder.embed_utterance(wav)
    speaker_embedding_history.append(embedding)

    # Apply clustering if enough embeddings are available
    speaker_label = cluster_speakers()
    if speaker_label is not None:
        identity = f"Speaker_{speaker_label + 1}"  # Assign a cluster ID
        known_speakers[identity] = embedding
    else:
        identity = "Unknown"

    return identity

def log_transcript(speaker, text):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line_with_speaker = f"[{timestamp}] {speaker}: {text}"
    line_plain = f"{text}"

    print(line_with_speaker)

    with transcript_lock:
        transcript_lines_speaker.append(line_with_speaker)
        transcript_lines_plain.append(line_plain)
        write_buffer.append((line_with_speaker, line_plain))

def process_chunk(audio_frames):
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
                # Correctly passing all required arguments
                speaker = identify_speaker(audio_frames, encoder, known_speakers, speaker_threshold, samplerate)
                log_transcript(speaker, text)

    except Exception as e:
        print(f"[❌ Whisper Error] {e}")

    end = time.time()
    latency_data.append(end - start)

def audio_thread():
    buffer = []
    while True:
        try:
            with sd.RawInputStream(samplerate=samplerate, blocksize=blocksize,
                                   device=device, dtype='int16', channels=1,
                                   callback=callback):
                while True:
                    data = q.get()
                    audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    buffer.append(audio_np)

                    if len(buffer) >= 3:
                        executor.submit(process_chunk, buffer[-3:])
                        buffer = buffer[-1:]
        except Exception as e:
            print(f"[❌ Audio Thread Error] {e} on device={device}")
            time.sleep(3)

def periodic_writer():
    while True:
        try:
            time.sleep(5)
            with transcript_lock:
                if write_buffer:
                    with open(speaker_transcript_path, "a", encoding="utf-8") as f1, \
                         open(plain_transcript_path, "a", encoding="utf-8") as f2:
                        for speaker_line, plain_line in write_buffer:
                            f1.write(speaker_line + "\n")
                            f2.write(plain_line + "\n")
                    write_buffer.clear()
        except Exception as e:
            print(f"[❌ Writer Thread Error] {e}")

def performance_monitor():
    while True:
        cpu = psutil.cpu_percent(interval=5)
        mem = psutil.virtual_memory().percent
        print(f"[📊 Performance] CPU: {cpu:.1f}% | Memory: {mem:.1f}%")

def start_background_tasks():
    threading.Thread(target=audio_thread, daemon=True).start()
    threading.Thread(target=periodic_writer, daemon=True).start()
    threading.Thread(target=performance_monitor, daemon=True).start()

# === FASTAPI ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/transcript")
async def get_transcript(mode: str = "plain"):
    with transcript_lock:
        if mode == "plain":
            return list(transcript_lines_plain)
        elif mode == "speaker":
            return list(transcript_lines_speaker)
        return []

@app.get("/status")
async def status():
    with transcript_lock:
        return {
            "known_speakers": list(known_speakers.keys()),
            "total_lines_speaker": len(transcript_lines_speaker),
            "total_lines_plain": len(transcript_lines_plain),
            "write_buffer_size": len(write_buffer),
            "queue_size": q.qsize(),
            "active_threads": threading.active_count(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "avg_latency_sec": round(np.mean(latency_data), 3) if latency_data else None,
            "max_latency_sec": round(np.max(latency_data), 3) if latency_data else None
        }

@app.get("/")
async def index():
    return {"message": "FastAPI is running. Use /transcript and /status endpoints."}

# === MAIN ENTRY POINT ===
if __name__ == "__main__":
    print(f"🎧 Using device: {device} | Torch device: {device_type}")
    print(f"📁 Transcripts will be saved to: {transcript_dir}")
    print("🚀 FastAPI server running at http://localhost:9575")
    start_background_tasks()
    uvicorn.run(app, host="0.0.0.0", port=9575, log_level="info")
