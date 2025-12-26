import gradio as gr
from core.dataset import load_jsonl_dataset
from core.summarizer_hf import summarize_text
from core.auto_qa_hf import generate_questions
from core.qa_hf import answer_question

DATASET_PATH = "datasets/my_dataset.jsonl"
docs = load_jsonl_dataset(DATASET_PATH)

doc_map = {f"{d['id']} - {d['title']}": d["text"] for d in docs}
doc_choices = ["-- none --"] + list(doc_map.keys())

def load_doc(choice):
    if choice == "-- none --":
        return ""
    return doc_map[choice]

def revision_mode(text, n_questions):
    # 1) summary (short)
    summary = summarize_text(text)

    # 2) generate questions FROM SUMMARY (revision style)
    questions = generate_questions(summary, n_questions=int(n_questions))

    # 3) answer questions FROM ORIGINAL TEXT (more accurate)
    qa_text = ""
    for i, q in enumerate(questions, start=1):
        ans = answer_question(text, q)
        qa_text += f"Q{i}: {q}\nA{i}: {ans}\n\n"

    return summary, qa_text

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 AI Revision App (Summary → Auto Questions → Answers)")
    gr.Markdown("### ✅ One-click revision: Generate short summary + automatic Q&A from summary")

    with gr.Row():
        choice = gr.Dropdown(doc_choices, label="📚 Choose Dataset Document")
        load_btn = gr.Button("Load Document")

    text = gr.Textbox(label="📝 Text Input", lines=10, placeholder="Paste your text here or load from dataset...")
    load_btn.click(load_doc, inputs=choice, outputs=text)

    n_questions = gr.Slider(3, 10, value=5, step=1, label="Number of Revision Questions")

    run_btn = gr.Button("🚀 Generate Revision (Summary + Auto Q&A)")

    summary_out = gr.Textbox(label="✅ Ultra Short Summary", lines=4)
    qa_out = gr.Textbox(label="✅ Auto Revision Questions & Answers", lines=14)

    run_btn.click(revision_mode, inputs=[text, n_questions], outputs=[summary_out, qa_out])

demo.launch(share=True)
