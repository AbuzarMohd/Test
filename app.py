# Model download must happen first
from download_models import download_models  # noqa: E402

# ────────── then heavy imports ──────────────────────────
import streamlit as st
from pipelines import text_distilbert as txt
from pipelines import voice_mfcc      as voc
from pipelines import face_fer        as fac
from pipelines import fuse
from brain     import llama_cpp_reply as bot
from brain     import memory
from components.audio_rec  import audio_recorder
from components.mood_chart import draw_chart

st.set_page_config("🧬 Emotion Mirror (Py 3.13 CPU)", layout="wide")

modal_logits: dict[str, list] = {}
mem = memory.ChatMemory()

st.title("🧬 Emotion Mirror – Reflect & Chat")

# Layout columns
chat_col, media_col = st.columns([3, 2])

# ---- TEXT --------------------------------------------------
with chat_col:
    utext = st.chat_input("What’s on your mind?")
    if utext:
        lbl, prob = txt.detect(utext)
        mem.add("user", utext)
        st.chat_message("user").write(utext)
        modal_logits["text"] = prob

# ---- VOICE -------------------------------------------------
with media_col:
    wav = audio_recorder("🎙️ Record voice")
    if wav:
        vlab, vprob = voc.detect(wav)
        st.success(f"Voice emotion → {vlab}")
        modal_logits["voice"] = vprob

# ---- FACE --------------------------------------------------
with media_col:
    snap = st.camera_input("📸 Webcam photo")
    if snap is not None and st.button("Analyse face"):
        flab, fprob = fac.detect(snap.getvalue())
        st.success(f"Face emotion → {flab}")
        modal_logits["face"] = fprob

# ---- REPLY -------------------------------------------------
if mem.last_is_user():
    idx, fused = fuse.fuse(modal_logits)
    emo = fuse.LABELS[idx]
    ans = bot.reply(mem.history, emo)
    mem.add("ai", ans)
    st.chat_message("assistant").markdown(ans)

# ---- Mood chart -------------------------------------------
with st.expander("📊 Mood trend"):
    draw_chart(mem.moodlog)
