import io
import os
import re
import string
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document
from sklearn.pipeline import Pipeline

# ---------- File readers ----------
def read_pdf(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for p in reader.pages:
        text += p.extract_text() or ""
    return text

def read_docx(file_bytes):
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs])

def read_txt(file_bytes):
    try:
        return file_bytes.decode('utf-8', errors='ignore')
    except Exception:
        return str(file_bytes)

def extract_text_from_file(file_name, file_bytes):
    ext = os.path.splitext(file_name)[1].lower()
    if ext == ".pdf":
        return read_pdf(file_bytes)
    elif ext in (".docx", ".doc"):
        return read_docx(file_bytes)
    elif ext in (".txt", ".rtf"):
        return read_txt(file_bytes)
    else:
        return ""

def extract_zip_bytes(zip_bytes):
    texts = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = info.filename
            try:
                b = z.read(name)
                txt = extract_text_from_file(name, b)
                if txt:
                    texts[name] = txt
            except Exception:
                continue
    return texts

# ---------- Cleaning ----------
def remove_html(text):
    return BeautifulSoup(text or "", "html.parser").get_text(separator=" ")

def remove_urls_emails(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    return text

def remove_punc_numbers(text):
    text = re.sub(r'\d+', '', text)
    return text.translate(str.maketrans('', '', string.punctuation))

DEFAULT_STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been before being below between both
but by could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has
hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm
i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or
other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that
that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through
to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's
which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

def clean_text(text):
    text = remove_html(text)
    text = text.lower()
    text = remove_urls_emails(text)
    text = remove_punc_numbers(text)
    tokens = [t.strip() for t in text.split() if t.strip() and t not in DEFAULT_STOPWORDS]
    return " ".join(tokens)

# ---------- Skills extractor (optional) ----------
SKILLS = [
    'react','reactjs','javascript','html','css','python','django','flask','node','nodejs',
    'sql','mysql','postgresql','mongodb','aws','azure','docker','kubernetes','git'
]
def extract_skills(text, skills=SKILLS):
    t = " " + (text or "") + " "
    found = [s for s in skills if f" {s} " in t or s in t]
    return sorted(set(found))

# ---------- Model selection / loading ----------
@st.cache_resource
def pick_model(joblib_paths=("best_resume_model.joblib","best_model.joblib")):
    # 1) If running inside the same notebook/session, prefer gs.best_estimator_ then pipeline or compose vectorizer+model
    try:
        g = globals()
        # prefer GridSearchCV .best_estimator_
        if 'gs' in g and hasattr(g['gs'], "best_estimator_"):
            return g['gs'].best_estimator_, "gs.best_estimator_"
        if 'pipeline' in g and isinstance(g['pipeline'], Pipeline):
            return g['pipeline'], "pipeline"
        # if raw classifier + vectorizer exist, compose a pipeline
        if 'vectorizer' in g and 'model' in g:
            try:
                composed = Pipeline([('tfidf', g['vectorizer']), ('clf', g['model'])])
                return composed, "composed(vectorizer+model)"
            except Exception:
                pass
    except Exception:
        pass
    # 2) Fallback: try load joblib model files from disk
    for p in joblib_paths:
        if os.path.exists(p):
            try:
                m = joblib.load(p)
                return m, f"joblib:{p}"
            except Exception:
                continue
    return None, None

model, model_source = pick_model()

def get_proba_or_scores(estimator, texts):
    try:
        proba = estimator.predict_proba(texts)
        return proba, estimator.classes_
    except Exception:
        try:
            scores = estimator.decision_function(texts)
            if scores.ndim == 1:
                scores = np.vstack([-scores, scores]).T
            exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            proba = exp / exp.sum(axis=1, keepdims=True)
            classes = estimator.classes_ if hasattr(estimator, 'classes_') else np.arange(proba.shape[1])
            return proba, classes
        except Exception:
            preds = estimator.predict(texts)
            classes = estimator.classes_ if hasattr(estimator, 'classes_') else np.unique(preds)
            proba = np.zeros((len(texts), len(classes)))
            cls_to_idx = {c:i for i,c in enumerate(classes)}
            for i,p in enumerate(preds):
                proba[i, cls_to_idx[p]] = 1.0
            return proba, classes

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Resume Classifier", layout="wide")
st.title("Resume Classifier (Notebook / Joblib aware)")

col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader("Upload resume (pdf/docx/txt) or zip", type=['pdf','docx','doc','txt','zip'])
    manual_text = st.text_area("Or paste resume text here", height=200)
    run_btn = st.button("Classify")
with col2:
    st.header("Model")
    if model is None:
        st.error("No model found in session or joblib. Place trained pipeline or joblib model next to app.")
    else:
        st.success(f"Loaded model ({model_source}): {type(model).__name__}")
        if hasattr(model, 'named_steps'):
            st.write("Pipeline steps:", list(model.named_steps.keys()))
    st.markdown("Top skills (searches):")
    st.write(", ".join(SKILLS))

def process_texts(texts_dict):
    rows = []
    texts = []
    fnames = []
    for fname, raw in texts_dict.items():
        cleaned = clean_text(raw)
        skills = extract_skills(cleaned)
        fnames.append(fname)
        texts.append(cleaned)
        rows.append({'file': fname, 'cleaned': cleaned[:1000], 'skills': ", ".join(skills)})
    if model is None:
        return pd.DataFrame(rows)
    proba, classes = get_proba_or_scores(model, texts)
    for i, r in enumerate(rows):
        pred_idx = int(np.argmax(proba[i]))
        r['predicted'] = classes[pred_idx]
        top_n = min(3, proba.shape[1])
        inds = np.argsort(proba[i])[::-1][:top_n]
        r['top_probs'] = ", ".join([f"{classes[j]}:{proba[i,j]:.3f}" for j in inds])
    return pd.DataFrame(rows)

if run_btn:
    texts_dict = {}
    if uploaded is not None:
        b = uploaded.read()
        if uploaded.type == "application/zip" or uploaded.name.lower().endswith(".zip"):
            texts_dict = extract_zip_bytes(b)
            if not texts_dict:
                st.warning("No readable resumes found inside ZIP.")
        else:
            txt = extract_text_from_file(uploaded.name, b)
            if not txt:
                st.warning("Could not extract text from uploaded file.")
            texts_dict[uploaded.name] = txt
    if manual_text:
        texts_dict["manual_input.txt"] = manual_text

    if not texts_dict:
        st.info("Please upload or paste text.")
    else:
        out = process_texts(texts_dict)
        st.subheader("Results")
        cols = ['file','predicted','top_probs','skills']
        present = [c for c in cols if c in out.columns]
        st.dataframe(out[present])
        if len(out) == 1:
            r = out.iloc[0]
            st.subheader("Detail")
            st.write("Predicted:", r.get('predicted',''))
            st.write("Top probs:", r.get('top_probs',''))
            st.write("Skills:", r.get('skills',''))
            st.text_area("Cleaned text", r.get('cleaned',''), height=300)

st.markdown("---")
st.write("Notes: this app tries to pick a best model from the running notebook (gs.best_estimator_, pipeline) or load a joblib model file. For production, save the full preprocessing+classifier pipeline with joblib and place it next to this app.")
