import gradio as gr
from core.dataset import load_jsonl_dataset
from core.summarizer_hf import summarize_text
from core.qa_hf import answer_question

DATASET_PATH = "datasets/my_dataset.jsonl"
docs = load_jsonl_dataset(DATASET_PATH)

doc_map = {f"{d['id']} - {d['title']}": d["text"] for d in docs}
doc_choices = ["-- none --"] + list(doc_map.keys())

def load_doc(choice):
    if choice == "-- none --":
        return ""
    return doc_map[choice]

def summarize_fn(text):
    return summarize_text(text)

def answer_fn(text, question):
    return answer_question(text, question)

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 AI Summarize + Q&A (Free)")

    with gr.Row():
        choice = gr.Dropdown(doc_choices, label="📚 Dataset document")
        load_btn = gr.Button("Load")
    text = gr.Textbox(label="📝 Text", lines=10)

    load_btn.click(load_doc, inputs=choice, outputs=text)

    with gr.Row():
        sum_btn = gr.Button("✨ Summarize")
        sum_out = gr.Textbox(label="✅ Summary", lines=6)

    sum_btn.click(summarize_fn, inputs=text, outputs=sum_out)

    question = gr.Textbox(label="❓ Question")
    ans_btn = gr.Button("🤖 Answer")
    ans_out = gr.Textbox(label="✅ Answer", lines=4)

    ans_btn.click(answer_fn, inputs=[text, question], outputs=ans_out)

demo.launch(share=True)
