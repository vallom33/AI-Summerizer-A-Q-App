import streamlit as st
from core.dataset import load_jsonl_dataset
from core.summarizer_hf import summarize_text
from core.auto_qa_hf import generate_questions
from core.qa_hf import answer_question_with_score

DATASET_PATH = "datasets/my_dataset.jsonl"
MIN_Q = 3

# ---------- Page config ----------
st.set_page_config(
    page_title="AI Summarizer + Auto Q&A",
    page_icon="🧠",
    layout="wide"
)

# ---------- Dark style ----------
st.markdown("""
<style>
    .stApp {background: #0b1220; color: #e6eefc;}
    h1, h2, h3, p, label {color:#e6eefc !important;}
    .block-container {padding-top: 2rem;}
    textarea {background:#0f1a30 !important; color:#e6eefc !important;}
    .stSelectbox, .stRadio {background:#0f1a30;}
    .stButton>button {background:#2563eb; color:white; border-radius:10px; padding:0.6rem 1rem;}
    .card {background:#0f1a30; border:1px solid #1f2a44; border-radius:14px; padding:14px; margin-bottom:12px;}
    .small {opacity:0.85;}
</style>
""", unsafe_allow_html=True)

# ---------- Load dataset ----------
try:
    docs = load_jsonl_dataset(DATASET_PATH)
except:
    docs = []

doc_map = {f"{d['id']} - {d['title']}": d["text"] for d in docs}
doc_choices = ["-- none --"] + list(doc_map.keys())

def generic_questions(lang: str):
    if lang == "fr":
        return ["De quoi parle le texte ?", "Quels sont les points importants ?", "Pourquoi cela est-il important ?",
                "Comment cela fonctionne ?", "Quel est le résultat ?"]
    if lang == "ar":
        return ["ما موضوع النص؟", "ما أهم النقاط؟", "لماذا هذا مهم؟", "كيف يحدث ذلك؟", "ما النتيجة؟"]
    return ["What is the text about?", "What are the key points?", "Why is it important?",
            "How does it work?", "What is the outcome?"]

def run_revision(text: str, lang: str):
    text = (text or "").strip()
    if len(text) < 80:
        msg = "Texte trop court." if lang == "fr" else ("النص قصير جدًا." if lang == "ar" else "Text too short.")
        return msg, []

    summary = summarize_text(text, lang=lang)

    candidates = generate_questions(text, lang=lang, n_questions=80)
    scored = []
    for q in candidates:
        res = answer_question_with_score(text, q)
        ans, score = res["answer"], res["score"]
        if score >= 0.10 and len(ans) >= 2:
            scored.append((score, q, ans))

    if len(scored) < MIN_Q:
        candidates2 = generate_questions(summary, lang=lang, n_questions=80)
        for q in candidates2:
            res = answer_question_with_score(text, q)
            ans, score = res["answer"], res["score"]
            if score >= 0.08 and len(ans) >= 2:
                scored.append((score, q, ans))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:MIN_Q]

    if len(top) < MIN_Q:
        for q in generic_questions(lang):
            res = answer_question_with_score(text, q)
            ans, score = res["answer"], res["score"]
            if score >= 0.03 and len(ans) >= 2:
                top.append((score, q, ans))
            if len(top) >= MIN_Q:
                break

    return summary, top

# ---------- UI ----------
st.title("🧠 AI Summarizer + Auto Q&A")
st.write("**EN / FR / AR** — Summary + revision questions with answers.")

col1, col2 = st.columns([1, 1])

with col1:
    lang = st.radio("Language / Langue / اللغة", ["en", "fr", "ar"], horizontal=True)
    choice = st.selectbox("📚 Dataset Document", doc_choices)

    if st.button("Load selected document"):
        st.session_state["text"] = doc_map.get(choice, "")

    text = st.text_area("📝 Text Input", height=240, value=st.session_state.get("text", ""))

    run = st.button("🚀 Generate (Summary + Auto Q&A)")

with col2:
    st.markdown('<div class="card"><h3>✅ Output</h3><p class="small">Summary + Q&A will appear here.</p></div>', unsafe_allow_html=True)

    if run:
        summary, qa_pairs = run_revision(text, lang)

        st.markdown('<div class="card"><h3>✅ Summary / Résumé / الملخص</h3></div>', unsafe_allow_html=True)
        st.write(summary)

        st.markdown('<div class="card"><h3>✅ Q&A / Questions-Réponses / أسئلة وأجوبة</h3></div>', unsafe_allow_html=True)
        if isinstance(qa_pairs, list) and qa_pairs:
            for i, (score, q, ans) in enumerate(qa_pairs, start=1):
                with st.expander(f"Q{i}: {q}"):
                    st.write(ans)
                    st.caption(f"Confidence: {score:.2f}")
        else:
            st.warning("Could not extract strong answers. Try longer / clearer text.")
