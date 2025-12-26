from transformers import pipeline

_qg = pipeline(
    "text2text-generation",
    model="iarfmoose/t5-base-question-generator",
    device=0
)

def generate_questions(text: str, n_questions: int = 5):
    text = text.strip()
    if len(text) < 40:
        return ["What is the main idea?"]

    prompt = f"generate questions: {text}"
    out = _qg(prompt, max_length=256, do_sample=False)
    gen = out[0]["generated_text"]

    # split by ?
    raw = [q.strip() for q in gen.split("?") if q.strip()]
    qs = [q + "?" for q in raw]

    # clean duplicates
    final = []
    seen = set()
    for q in qs:
        q_norm = q.lower()
        if q_norm not in seen and len(q) > 5:
            final.append(q)
            seen.add(q_norm)
        if len(final) >= n_questions:
            break

    # fallback if model returns nothing
    if not final:
        final = [
            "What is the main topic?",
            "What are the key points?",
            "Why is it important?"
        ][:n_questions]

    return final
