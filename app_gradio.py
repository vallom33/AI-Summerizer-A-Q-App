
import gradio as gr
from core.dataset import load_jsonl_dataset
from core.summarizer_hf import summarize_text
from core.auto_qa_hf import generate_questions
from core.qa_hf import answer_question_with_score

DATASET_PATH = "datasets/my_dataset.jsonl"
docs = load_jsonl_dataset(DATASET_PATH)

doc_map = {f"{d['id']} - {d['title']}": d["text"] for d in docs}
doc_choices = ["-- none --"] + list(doc_map.keys())

def load_doc(choice):
    if choice == "-- none --":
        return ""
    return doc_map[choice]



def revision_mode(text, n_questions):
    text = (text or "").strip()
    if len(text) < 80:
        return "Text too short.", "Please provide a longer text."

    target = int(n_questions)

    # ✅ Generate MANY questions to increase chance of good ones
    questions = generate_questions(text, n_questions=target * 6)

    scored_pairs = []
    for q in questions:
        res = answer_question_with_score(text, q)
        ans, score = res["answer"], res["score"]

        # ✅ filter out weak/empty answers
        if score < 0.20 or len(ans) < 2:
            continue

        scored_pairs.append((score, q, ans))

    # ✅ Sort by confidence score and keep top N
    scored_pairs.sort(key=lambda x: x[0], reverse=True)
    top_pairs = scored_pairs[:target]

    summary = summarize_text(text)

    # ✅ If still empty, show message (NO hardcoded topic questions)
    if not top_pairs:
        return summary, "Could not generate strong Q&A pairs. Try a longer text or increase text clarity."

    qa_text = ""
    for i, (score, q, ans) in enumerate(top_pairs, start=1):
        qa_text += f"Q{i}: {q}\nA{i}: {ans}\n(Confidence: {score:.2f})\n\n"

    return summary, qa_text


with gr.Blocks() as demo:
    gr.Markdown("# 🧠 AI Revision App (Summary + Auto Q&A)")
    gr.Markdown("✅ Generate a short summary + revision questions with answers extracted from the text.")

    with gr.Row():
        choice = gr.Dropdown(doc_choices, label="📚 Choose Dataset Document")
        load_btn = gr.Button("Load Document")

    text = gr.Textbox(label="📝 Text Input", lines=10, placeholder="Paste your text or load from dataset...")
    load_btn.click(load_doc, inputs=choice, outputs=text)

    n_questions = gr.Slider(3, 10, value=5, step=1, label="Number of Revision Questions")
    run_btn = gr.Button("🚀 Generate Revision (Summary + Auto Q&A)")

    summary_out = gr.Textbox(label="✅ Ultra Short Summary", lines=4)
    qa_out = gr.Textbox(label="✅ Auto Revision Questions & Answers", lines=14)

    run_btn.click(revision_mode, inputs=[text, n_questions], outputs=[summary_out, qa_out])

demo.launch(share=True)
