from llama_cpp import Llama
import multiprocessing, functools, os

_MODEL = "models/tinyllama-1.1B-chat.Q4_K_M.gguf"

@functools.lru_cache(maxsize=1)
def _llm():
    n_threads = min(4, multiprocessing.cpu_count())
    return Llama(model_path=_MODEL, n_ctx=1536, n_threads=n_threads)

def reply(hist, emo):
    past = "\n".join(f"{r.upper()}: {t}" for r, t in hist[-6:])
    prompt = (f"[INST]You are Emotion Mirror, supportive AI.\n"
              f"Emotion detected: {emo}\n{past}\nAI:[/INST]")
    out = _llm()(prompt, max_tokens=140, stop=["[/INST]"])
    return out["choices"][0]["text"].strip()
