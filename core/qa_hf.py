from transformers import pipeline

_qa = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    device=0
)

def answer_question(context: str, question: str) -> str:
    context = context.strip()
    question = question.strip()

    if not context:
        return "لا يوجد نص للإجابة عليه."
    if not question:
        return "الرجاء إدخال سؤال."

    result = _qa(question=question, context=context)
    answer = result.get("answer", "").strip()
    score = float(result.get("score", 0.0))

    if not answer or score < 0.20:
        return "لا أستطيع الجواب لأن المعلومة غير واضحة أو غير موجودة في النص."

    return f"{answer}\n\n(Confidence: {score:.2f})"
