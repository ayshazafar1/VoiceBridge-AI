from __future__ import annotations
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ================= CONFIG =================
@dataclass (frozen=True)
class AppConfig:
    whisper_model: str
    whisper_language: str
    whisper_device: str
    whisper_compute_type: str
    whisper_cpu_threads: int

    groq_api_key: str
    groq_model_id: str
    system_prompt: str

    tts_engine: str
    gtts_lang: str
    piper_model_path: str
    piper_config_path: str


def load_config() -> AppConfig:
    return AppConfig(
        whisper_model=os.getenv("WHISPER_MODEL", "base.en"),
        whisper_language=os.getenv("WHISPER_LANGUAGE", "en"),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        whisper_cpu_threads=int(os.getenv("WHISPER_CPU_THREADS", 4)),

        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model_id=os.getenv("GROQ_MODEL_ID", "llama-3.1-8b-instant"),
        system_prompt=os.getenv(
            "SYSTEM_PROMPT",
            "You are a helpful voice assistant. Keep replies short."
        ),

        tts_engine="gtts",
        gtts_lang="en",
        piper_model_path=os.getenv("PIPER_MODEL_PATH", ""),
        piper_config_path=os.getenv("PIPER_CONFIG_PATH", "")
    )


CFG = load_config()

# ================= UI CONFIG =================
st.set_page_config(page_title="VoiceBridge AI", layout="centered")

# ======= SHASHSKAY CSS =======
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1f1c2c, #928dab);
    color: white;
}
h1 {
    text-align: center;
}
[data-testid="stChatMessage"] {
    border-radius: 18px;
    padding: 12px;
    margin-bottom: 10px;
    backdrop-filter: blur(10px);
    background: rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.4);
}
.stButton button {
    border-radius: 12px;
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<h1>🎙️ VoiceBridge AI</h1>
<p style='text-align:center;'>Talk naturally. Get instant AI voice replies.</p>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    CFG.tts_engine = st.selectbox("Voice Engine", ["gtts", "piper"])
    lang = st.selectbox("Language", ["en", "ur"])

    CFG.whisper_language = lang
    CFG.gtts_lang = lang

    if st.button("🗑 Clear Chat"):
        st.session_state.clear()
        st.rerun()

# ================= ASR =================
@st.cache_resource
def get_whisper():
    from faster_whisper import WhisperModel
    return WhisperModel(
        CFG.whisper_model,
        device=CFG.whisper_device,
        compute_type=CFG.whisper_compute_type,
        cpu_threads=CFG.whisper_cpu_threads
    )


def transcribe(audio_bytes: bytes) -> str:
    model = get_whisper()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        path = f.name

    segments, _ = model.transcribe(path, language=CFG.whisper_language, vad_filter=True)
    os.remove(path)

    return "".join([seg.text for seg in segments]).strip()


# ================= LLM =================
def offline_reply(text: str) -> str:
    return f"(Offline Mode)\nYou said: {text}"


def groq_reply(messages: List[Dict[str, str]]) -> str:
    import requests

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {CFG.groq_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": CFG.groq_model_id,
        "messages": messages,
        "temperature": 0.4
    }

    r = requests.post(url, headers=headers, json=payload)
    return r.json()["choices"][0]["message"]["content"]


def generate_reply(user_text, history):
    if not CFG.groq_api_key:
        return offline_reply(user_text)

    messages = [{"role": "system", "content": CFG.system_prompt}]
    messages += history[-6:]
    messages.append({"role": "user", "content": user_text})

    return groq_reply(messages)


# ================= TTS =================
def gtts_tts(text):
    from gtts import gTTS

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        path = f.name

    gTTS(text=text, lang=CFG.gtts_lang).save(path)

    audio = open(path, "rb").read()
    os.remove(path)
    return audio, "audio/mpeg"


def piper_tts(text):
    from piper import PiperVoice
    import wave

    if not CFG.piper_model_path:
        return gtts_tts("Piper not configured")

    voice = PiperVoice.load(CFG.piper_model_path, CFG.piper_config_path)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name

    with wave.open(path, "wb") as wf:
        voice.synthesize_wav(text, wf)

    audio = open(path, "rb").read()
    os.remove(path)
    return audio, "audio/wav"


def text_to_speech(text):
    if CFG.tts_engine == "piper":
        return piper_tts(text)
    return gtts_tts(text)


# ================= STATE =================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ================= CHAT DISPLAY =================
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ================= AUDIO INPUT =================
audio = st.audio_input("🎤 Speak")

if audio:
    audio_bytes = audio.getvalue()

    if "last_audio" not in st.session_state or st.session_state.last_audio != audio_bytes:
        st.session_state.last_audio = audio_bytes

        # ASR
        start = time.time()
        text = transcribe(audio_bytes)
        asr_time = time.time() - start

        if not text or len(text) < 2:
            st.warning("No speech detected.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": text})

            with st.chat_message("user"):
                st.write(text)

            # LLM
            start = time.time()
            reply = generate_reply(text, st.session_state.chat_history)
            llm_time = time.time() - start

            st.session_state.chat_history.append({"role": "assistant", "content": reply})

            # TTS
            start = time.time()
            audio_bytes, mime = text_to_speech(reply)
            tts_time = time.time() - start

            # Assistant UI
            with st.chat_message("assistant"):
                st.markdown(f"**🤖 AI:** {reply}")
                st.audio(audio_bytes, format=mime)

                st.caption(
                    f"⚡ ASR: {asr_time:.2f}s | LLM: {llm_time:.2f}s | TTS: {tts_time:.2f}s"
                )