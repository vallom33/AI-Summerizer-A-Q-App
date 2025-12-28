from transformers import pipeline

_qa = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    device=0
)

def answer_question_with_score(context: str, question: str):
    context = context.strip()
    question = question.strip()
    if not context or not question:
        return {"answer": "", "score": 0.0}

    result = _qa(question=question, context=context)
    answer = (result.get("answer") or "").strip()
    score = float(result.get("score", 0.0))

    # if model returns empty / nonsense
    if not answer:
        score = 0.0

    return {"answer": answer, "score": score}
