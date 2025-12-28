from transformers import pipeline

# Multilingual summarizer
_summarizer = pipeline(
    "text2text-generation",
    model="csebuetnlp/mT5_multilingual_XLSum"
)

def summarize_text(text: str, lang: str = "en") -> str:
    text = (text or "").strip()
    if len(text) < 80:
        return "Texte trop court pour résumer." if lang == "fr" else "Text too short to summarize."

    prompt = f"summarize: {text}"
    out = _summarizer(prompt, max_length=80, do_sample=False)
    return out[0]["generated_text"].strip()
