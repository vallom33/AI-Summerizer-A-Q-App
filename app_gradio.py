import gradio as gr
from core.dataset import load_jsonl_dataset
from core.summarizer_hf import summarize_text
from core.auto_qa_hf import auto_revision_qa

DATASET_PATH = "datasets/my_dataset.jsonl"
docs = load_jsonl_dataset(DATASET_PATH)

doc_map = {f"{d['id']} - {d['title']}": d["text"] for d in docs}
doc_choices = ["-- none --"] + list(doc_map.keys())

def load_doc(choice):
    if choice == "-- none --":
        return ""
    return doc_map[choice]

def run_revision(text):
    summary = summarize_text(text)
    qa_pairs = auto_revision_qa(text, n_questions=5)

    qa_text = ""
    for i, (q, a) in enumerate(qa_pairs, start=1):
        qa_text += f"Q{i}: {q}\nA{i}: {a}\n\n"

    return summary, qa_text

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 AI Revision App (Summarize + Auto Q&A)")

    with gr.Row():
        choice = gr.Dropdown(doc_choices, label="📚 Dataset document")
        load_btn = gr.Button("Load Selected Doc")

    text = gr.Textbox(label="📝 Text", lines=10)

    load_btn.click(load_doc, inputs=choice, outputs=text)

    run_btn = gr.Button("🚀 Generate Summary + Auto Q&A (Revision)")
    summary_out = gr.Textbox(label="✅ Ultra Short Summary", lines=4)
    qa_out = gr.Textbox(label="✅ Revision Questions & Answers", lines=12)

    run_btn.click(run_revision, inputs=text, outputs=[summary_out, qa_out])

demo.launch(share=True)
