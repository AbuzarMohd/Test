from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, functools, numpy as np

_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

@functools.lru_cache(maxsize=1)
def _load():
    tok = AutoTokenizer.from_pretrained(_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(_MODEL)
    mdl.eval()
    return tok, mdl

def detect(text: str):
    tok, mdl = _load()
    with torch.no_grad():
        out = mdl(**tok(text, return_tensors="pt", truncation=True)).logits[0]
    probs = torch.softmax(out, dim=0).cpu().numpy()
    label = "positive" if probs[1] > probs[0] else "negative"
    return label, probs
