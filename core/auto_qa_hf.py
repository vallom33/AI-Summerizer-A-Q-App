from transformers import pipeline

_qg = pipeline("text2text-generation", model="google/mt5-base")

def generate_questions(text: str, lang: str = "en", n_questions: int = 50):
    text = (text or "").strip()
    if len(text) < 80:
        return ["Quel est le sujet principal ?"] if lang == "fr" else ["What is the main topic?"]

    if lang == "fr":
        prompt = (
            "Génère 10 questions courtes de révision basées uniquement sur ce texte. "
            "Une question par ligne.\n\n"
            f"TEXTE:\n{text}"
        )
    else:
        prompt = (
            "Generate 10 short revision questions based only on this text. "
            "One question per line.\n\n"
            f"TEXT:\n{text}"
        )

    out = _qg(prompt, max_length=256, do_sample=False)[0]["generated_text"]
    lines = [l.strip() for l in out.split("\n") if l.strip()]

    questions = []
    for l in lines:
        l = l.lstrip("-• ").strip()
        if len(l) < 6:
            continue
        if not l.endswith("?"):
            l += "?"
        questions.append(l)

    # dedupe + limit
    seen = set()
    final = []
    for q in questions:
        qn = q.lower()
        if qn not in seen:
            final.append(q)
            seen.add(qn)
        if len(final) >= n_questions:
            break

    if not final:
        final = ["Quel est le sujet principal ?", "Quels sont les points importants ?", "Quelle est la conclusion ?"] if lang == "fr" else \
                ["What is the main topic?", "What are the key points?", "What is the conclusion?"]

    return final
