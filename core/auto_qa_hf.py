from transformers import pipeline

_qg = pipeline("text2text-generation", model="google/mt5-base")

def generate_questions(text: str, lang: str = "en", n_questions: int = 60):
    text = (text or "").strip()
    if len(text) < 80:
        if lang == "fr":
            return ["Quel est le sujet principal ?"]
        if lang == "ar":
            return ["ما موضوع النص؟"]
        return ["What is the main topic?"]

    if lang == "fr":
        prompt = (
            "Génère des questions de révision courtes basées uniquement sur ce texte. "
            "Une question par ligne.\n\n"
            f"TEXTE:\n{text}"
        )
    elif lang == "ar":
        prompt = (
            "أنشئ أسئلة مراجعة قصيرة اعتمادًا على هذا النص فقط. "
            "اكتب سؤالًا في كل سطر.\n\n"
            f"النص:\n{text}"
        )
    else:
        prompt = (
            "Generate short revision questions based only on this text. "
            "One question per line.\n\n"
            f"TEXT:\n{text}"
        )

    out = _qg(prompt, max_length=256, do_sample=False)[0]["generated_text"]
    lines = [l.strip() for l in out.split("\n") if l.strip()]

    qs = []
    for l in lines:
        l = l.lstrip("-• ").strip()
        if len(l) < 6:
            continue
        if not l.endswith("?") and not l.endswith("؟"):
            l += "؟" if lang == "ar" else "?"
        qs.append(l)

    # dedupe + limit
    seen = set()
    final = []
    for q in qs:
        qn = q.lower()
        if qn not in seen:
            final.append(q)
            seen.add(qn)
        if len(final) >= n_questions:
            break

    if not final:
        if lang == "fr":
            final = ["Quel est le sujet ?", "Quels sont les points clés ?", "Pourquoi est-ce important ?"]
        elif lang == "ar":
            final = ["ما موضوع النص؟", "ما أهم النقاط؟", "لماذا هذا مهم؟"]
        else:
            final = ["What is the topic?", "What are the key points?", "Why is it important?"]

    return final
