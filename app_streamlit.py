import streamlit as st
from core.dataset import load_jsonl_dataset
from core.summarizer_hf import summarize_text
from core.qa_hf import answer_question

st.set_page_config(page_title="AI Summarize + Q&A", layout="wide")
st.title("🧠 AI Summarize + Q&A (MVP)")

DATASET_PATH = "datasets/my_dataset.jsonl"

docs = []
try:
    docs = load_jsonl_dataset(DATASET_PATH)
except:
    docs = []

st.sidebar.header("📚 Dataset")
selected_doc = None

if docs:
    options = ["-- none --"] + [f"{d['id']} - {d['title']}" for d in docs]
    choice = st.sidebar.selectbox("اختر وثيقة", options)

    if choice != "-- none --":
        doc_id = choice.split(" - ")[0]
        selected_doc = next(d for d in docs if d["id"] == doc_id)
else:
    st.sidebar.info("ضع dataset في datasets/my_dataset.jsonl")

text = st.text_area("📝 أدخل النص أو اختر وثيقة من Dataset", height=250)

if selected_doc and st.sidebar.button("⬇️ Load selected document"):
    st.session_state["loaded_text"] = selected_doc["text"]

if "loaded_text" in st.session_state and not text.strip():
    text = st.session_state["loaded_text"]

col1, col2 = st.columns(2)

with col1:
    if st.button("✨ Summarize"):
        st.subheader("✅ Summary")
        st.write(summarize_text(text))

with col2:
    q = st.text_input("❓ Ask a question")
    if st.button("🤖 Answer"):
        st.subheader("✅ Answer")
        st.write(answer_question(text, q))
