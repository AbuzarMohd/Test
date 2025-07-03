import tempfile, joblib, os, numpy as np, soundfile as sf
from python_speech_features import mfcc

_SVM_PATH = "models/voice_svm_mfcc.joblib"
svm = joblib.load(_SVM_PATH) if os.path.isfile(_SVM_PATH) else None

def _extract_mfcc(wav_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(wav_bytes)
        path = f.name
    y, sr = sf.read(path)
    m = mfcc(y, sr, numcep=13)
    return np.mean(m, axis=0)

def detect(wav_bytes):
    if svm is None:
        return "unknown", np.zeros(2)
    feat = _extract_mfcc(wav_bytes)
    probs = svm.predict_proba([feat])[0]          # assumes 2 classes
    label = svm.classes_[int(np.argmax(probs))]
    return label, probs
