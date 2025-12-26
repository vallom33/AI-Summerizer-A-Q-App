from transformers import pipeline
from .qa_hf import answer_question

# Question generation model (T5)
_qg = pipeline(
    "text2text-generation",
    model="iarfmoose/t5-base-question-generator",
    device=0
)

def generate_questions(text: str, n_questions: int = 5):
    """
    Generates revision questions from text.
    """
    text = text.strip()
    if len(text) < 80:
        return ["The text is too short to generate questions."]

    prompt = f"generate questions: {text}"
    out = _qg(prompt, max_length=256, do_sample=False)

    generated = out[0]["generated_text"]

    # Split questions (model outputs multiple questions separated by ?)
    questions = []
    for part in generated.split("?"):
        q = part.strip()
        if q:
            questions.append(q + "?")

    # Remove duplicates + limit
    seen = set()
    final_qs = []
    for q in questions:
        if q.lower() not in seen:
            final_qs.append(q)
            seen.add(q.lower())
        if len(final_qs) >= n_questions:
            break

    return final_qs

def auto_revision_qa(text: str, n_questions: int = 5):
    """
    Generates questions and answers them from the same text.
    Returns list of (question, answer).
    """
    qs = generate_questions(text, n_questions=n_questions)
    qa_pairs = []
    for q in qs:
        ans = answer_question(text, q)
        qa_pairs.append((q, ans))
    return qa_pairs
