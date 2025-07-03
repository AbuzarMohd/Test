from fer import FER
import cv2, numpy as np, functools

@functools.lru_cache(maxsize=1)
def _det(): return FER(mtcnn=True)

def detect(img_bytes):
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    res = _det().detect_emotions(img)
    if not res:
        return "neutral", np.zeros(7)
    emo = res[0]["emotions"]
    lbl = max(emo, key=emo.get)
    return lbl, np.array(list(emo.values()))
