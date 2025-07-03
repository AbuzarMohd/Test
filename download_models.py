import os, requests, logging, streamlit as st

logging.basicConfig(level=logging.INFO)
CHUNK = 2**20                                 # 1 MB

def _stream(url, dst):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for c in r.iter_content(CHUNK):
                if c:
                    f.write(c)

def download_models():
    os.makedirs("models", exist_ok=True)
    assets = [
        ("https://huggingface.co/TheBloke/"
         "TinyLlama-1.1B-Chat-GGUF/resolve/main/"
         "tinyllama-1.1B-chat.Q4_K_M.gguf",
         "models/tinyllama-1.1B-chat.Q4_K_M.gguf",
         520),
        ("https://huggingface.co/robinjia/emo_mirror_assets/raw/main/"
         "voice_svm_mfcc.joblib",
         "models/voice_svm_mfcc.joblib",
         3),
    ]

    with st.spinner("⏬ Downloading TinyLLaMA & SVM (first run only)…"):
        for url, path, mb in assets:
            if not (os.path.exists(path) and os.path.getsize(path) > mb*0.9*2**20):
                logging.info(f"Downloading {os.path.basename(path)} …")
                _stream(url, path)
                logging.info("Done.")
            else:
                logging.info(f"{os.path.basename(path)} present.")

download_models()
