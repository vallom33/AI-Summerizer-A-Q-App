from transformers import pipeline

# Create once (global) for speed
_summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=0  # GPU in Colab
)

def summarize_text(text: str) -> str:
    text = text.strip()
    if len(text) < 80:
        return "النص قصير جدًا للتلخيص."

    # BART input length limit ~1024 tokens, so keep text reasonable
    out = _summarizer(
        text,
        max_length=160,
        min_length=40,
        do_sample=False
    )
    return out[0]["summary_text"]
