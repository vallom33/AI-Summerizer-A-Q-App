import gradio as gr
from core.dataset import load_jsonl_dataset
from core.summarizer_hf import summarize_text
from core.auto_qa_hf import generate_questions
from core.qa_hf import answer_question_with_score

DATASET_PATH = "datasets/my_dataset.jsonl"
MIN_Q = 3

docs = load_jsonl_dataset(DATASET_PATH)
doc_map = {f"{d['id']} - {d['title']}": d["text"] for d in docs}
doc_choices = ["-- none --"] + list(doc_map.keys())

def load_doc(choice):
    if choice == "-- none --":
        return ""
    return doc_map.get(choice, "")

def _generic_questions(lang: str):
    # General questions only (not topic-specific)
    if lang == "fr":
        return ["De quoi parle le texte ?", "Quels sont les points importants ?", "Pourquoi cela est-il important ?",
                "Comment cela fonctionne ?", "Quel est le résultat ?"]
    if lang == "ar":
        return ["ما موضوع النص؟", "ما أهم النقاط؟", "لماذا هذا مهم؟", "كيف يحدث ذلك؟", "ما النتيجة؟"]
    return ["What is the text about?", "What are the key points?", "Why is it important?",
            "How does it work?", "What is the outcome?"]

def revision_mode(text, lang):
    text = (text or "").strip()
    if len(text) < 80:
        msg = "Texte trop court." if lang == "fr" else ("النص قصير جدًا." if lang == "ar" else "Text too short.")
        return msg, msg

    summary = summarize_text(text, lang=lang)

    # 1) Generate MANY candidate questions
    candidates = generate_questions(text, lang=lang, n_questions=80)

    scored = []
    for q in candidates:
        res = answer_question_with_score(text, q)
        ans, score = res["answer"], res["score"]
        if score >= 0.10 and len(ans) >= 2:
            scored.append((score, q, ans))

    # 2) If not enough, try generating from summary
    if len(scored) < MIN_Q:
        candidates2 = generate_questions(summary, lang=lang, n_questions=80)
        for q in candidates2:
            res = answer_question_with_score(text, q)
            ans, score = res["answer"], res["score"]
            if score >= 0.08 and len(ans) >= 2:
                scored.append((score, q, ans))

    # 3) Rank and select
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:MIN_Q]

    # 4) Final: general intelligent questions (to guarantee 3 when possible)
    if len(top) < MIN_Q:
        for q in _generic_questions(lang):
            res = answer_question_with_score(text, q)
            ans, score = res["answer"], res["score"]
            if score >= 0.03 and len(ans) >= 2:
                top.append((score, q, ans))
            if len(top) >= MIN_Q:
                break

    qa_text = ""
    for i, (score, q, ans) in enumerate(top[:MIN_Q], start=1):
        qa_text += f"Q{i}: {q}\nA{i}: {ans}\n(Confidence: {score:.2f})\n\n"

    # If still empty (rare), show message but keep app stable
    if not qa_text.strip():
        qa_text = "Could not extract strong answers from this text. Try longer / clearer text."

    return summary, qa_text


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 AI Summarizer + Auto Q&A (EN/FR/AR)")
    gr.Markdown("Paste text or load from dataset → get summary + revision Q&A.")

    with gr.Row():
        lang = gr.Radio(["en", "fr", "ar"], value="en", label="Language / Langue / اللغة")
        choice = gr.Dropdown(doc_choices, label="📚 Dataset Document")
        load_btn = gr.Button("Load Doc")

    text = gr.Textbox(label="📝 Text Input", lines=10)
    load_btn.click(load_doc, inputs=choice, outputs=text)

    run_btn = gr.Button("🚀 Generate (Summary + Auto Q&A)")
    summary_out = gr.Textbox(label="✅ Summary / Résumé / الملخص", lines=4)
    qa_out = gr.Textbox(label="✅ Q&A / Questions-Réponses / أسئلة وأجوبة", lines=12)

    run_btn.click(revision_mode, inputs=[text, lang], outputs=[summary_out, qa_out])

demo.launch(share=True)
