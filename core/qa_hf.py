from transformers import pipeline

_qa = pipeline(
    "question-answering",
    model="deepset/xlm-roberta-large-squad2"
)

def answer_question_with_score(context: str, question: str):
    context = (context or "").strip()
    question = (question or "").strip()
    if not context or not question:
        return {"answer": "", "score": 0.0}

    res = _qa(question=question, context=context)
    ans = (res.get("answer") or "").strip()
    score = float(res.get("score", 0.0))

    if not ans:
        return {"answer": "", "score": 0.0}

    return {"answer": ans, "score": score}
