import streamlit as st
import spacy
import PyPDF2
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Config
st.set_page_config(page_title="AI Recruiter Pro", page_icon="👔", layout="wide")

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_md")

nlp = load_nlp()

# 2. Extraction Functions (Role, Experience, Education)
def extract_text(file):
    pdf_reader = PyPDF2.PdfReader(file)
    return " ".join([page.extract_text() for page in pdf_reader.pages])

def get_experience(text):
    exp_pattern = r"(\d+(?:\+)?\s*(?:years?|yrs?))"
    matches = re.findall(exp_pattern, text, re.IGNORECASE)
    return matches[0] if matches else "0-1 Years"

def get_education(text):
    edu_keywords = ["B.Tech", "M.Tech", "BCA", "MCA", "B.E", "BSc", "MSc", "MBA", "PhD", "Bachelor", "Master"]
    for word in edu_keywords:
        if word.lower() in text.lower():
            return word
    return "Undergraduate"

def get_job_role(text):
    lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 5]
    return lines[0] if lines else "Software Engineer"

# 3. UI Design
st.title("👔 AI Recruiter Pro: Smart Batch Screening")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("📋 Job Requirements")
    jd_input = st.text_area("Paste Job Description Here", height=300)

with col2:
    st.header("📤 Upload Candidate CVs")
    uploaded_files = st.file_uploader("Upload multiple PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("🚀 Analyze All Candidates") and uploaded_files and jd_input:
        results = []
        for file in uploaded_files:
            raw_text = extract_text(file)
            experience = get_experience(raw_text)
            education = get_education(raw_text)
            role = get_job_role(raw_text)
            
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([jd_input, raw_text])
            score = round(cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100, 1)
            
            jd_doc = nlp(jd_input.lower())
            res_doc = nlp(raw_text.lower())
            jd_skills = set([t.text for t in jd_doc if t.pos_ in ["NOUN", "PROPN"] and not t.is_stop])
            res_skills = set([t.text for t in res_doc if t.pos_ in ["NOUN", "PROPN"]])
            missing = list(jd_skills - res_skills)[:3]

            results.append({
                "Candidate": file.name,
                "Current Role": role,
                "Experience": experience,
                "Education": education,
                "Match Score (%)": score,
                "Missing Skills": ", ".join(missing) if score < 80 else "Skills Match!"
            })

        df = pd.DataFrame(results).sort_values(by="Match Score (%)", ascending=False)
        st.markdown("### 📊 Screening Dashboard")
        # Fixed the width setting here to stop the terminal error
        st.dataframe(df, width=1200, hide_index=True)
        st.download_button("📥 Download Report", df.to_csv(index=False), "screening_report.csv")

    elif not jd_input or not uploaded_files:
        st.info("💡 Paste a Job Description and upload CVs to begin.")