import streamlit as st, typing as t

def audio_recorder(label="🎙️ Record", pause_threshold=1.0) -> t.Optional[bytes]:
    try:
        from streamlit_audio_recorder import st_audiorec
        return st_audiorec(label=label, pause_threshold=pause_threshold)
    except ModuleNotFoundError:
        st.info("Recorder component missing; upload a WAV instead.")
        up = st.file_uploader("Upload WAV", type=["wav"])
        return up.read() if up else None
