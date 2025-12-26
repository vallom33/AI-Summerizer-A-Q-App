from transformers import pipeline

_summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=0
)

def summarize_text(text: str) -> str:
    text = text.strip()
    if len(text) < 80:
        return "النص قصير جدًا للتلخيص."

    # Ultra short summary
    out = _summarizer(
        text,
        max_length=60,   # كان 160
        min_length=20,   # كان 40
        do_sample=False
    )
    return out[0]["summary_text"]
