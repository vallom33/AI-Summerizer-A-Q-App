import gradio as gr
from core.dataset import load_jsonl_dataset
from core.summarizer_hf import summarize_text
from core.auto_qa_hf import generate_questions
from core.qa_hf import answer_question_with_score

DATASET_PATH = "datasets/my_dataset.jsonl"
MIN_QUESTIONS = 3

docs = load_jsonl_dataset(DATASET_PATH)
doc_map = {f"{d['id']} - {d['title']}": d["text"] for d in docs}
doc_choices = ["-- none --"] + list(doc_map.keys())

def load_doc(choice):
    if choice == "-- none --":
        return ""
    return doc_map.get(choice, "")

def revision_mode(text, lang):
    text = (text or "").strip()
    if len(text) < 80:
        msg = "Texte trop court." if lang == "fr" else "Text too short."
        return msg, msg

    summary = summarize_text(text, lang=lang)

    # generate many questions -> rank by QA confidence -> pick best 3
    candidates = generate_questions(text, lang=lang, n_questions=60)

    scored = []
    for q in candidates:
        res = answer_question_with_score(text, q)
        ans, score = res["answer"], res["score"]
        if score < 0.10 or len(ans) < 2:
            continue
        scored.append((score, q, ans))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:MIN_QUESTIONS]

    # second chance: generate from summary
    if len(top) < MIN_QUESTIONS:
        candidates2 = generate_questions(summary, lang=lang, n_questions=60)
        for q in candidates2:
            res = answer_question_with_score(text, q)
            ans, score = res["answer"], res["score"]
            if score < 0.08 or len(ans) < 2:
                continue
            scored.append((score, q, ans))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:MIN_QUESTIONS]

    # final fallback (GENERAL questions only)
    if len(top) < MIN_QUESTIONS:
        generic = (
            ["De quoi parle le texte ?", "Quel est le fait le plus important ?", "Quelle est la conclusion principale ?"]
            if lang == "fr"
            else ["What is the text about?", "What is the most important fact?", "What is the main conclusion?"]
        )
        for q in generic:
            res = answer_question_with_score(text, q)
            ans, score = res["answer"], res["score"]
            if score > 0.03 and len(ans) > 1:
                top.append((score, q, ans))

    qa_text = ""
    for i, (score, q, ans) in enumerate(top[:MIN_QUESTIONS], start=1):
        qa_text += f"Q{i}: {q}\nA{i}: {ans}\n(Confidence: {score:.2f})\n\n"

    return summary, qa_text


with gr.Blocks() as demo:
    gr.Markdown("# 🧠 AI Summarizer + Auto Q&A (EN/FR)")
    gr.Markdown("✅ Generates short summary + **3 automatic revision questions** with answers.")

    with gr.Row():
        lang = gr.Radio(["en", "fr"], value="en", label="Language / Langue")
        choice = gr.Dropdown(doc_choices, label="📚 Dataset Document")
        load_btn = gr.Button("Load Doc")

    text = gr.Textbox(label="📝 Text Input", lines=10)
    load_btn.click(load_doc, inputs=choice, outputs=text)

    run_btn = gr.Button("🚀 Generate (Summary + 3 Q&A)")
    summary_out = gr.Textbox(label="✅ Summary / Résumé", lines=4)
    qa_out = gr.Textbox(label="✅ Questions & Answers", lines=12)

    run_btn.click(revision_mode, inputs=[text, lang], outputs=[summary_out, qa_out])

demo.launch(share=True)
