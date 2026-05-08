import streamlit as st
import PyPDF2
import spacy
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP Engine
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_md")

nlp = load_nlp()

# --- PAGE CONFIG ---
st.set_page_config(page_title="TalentForce AI Pro", layout="wide", page_icon="🎯")

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stAlert { border-radius: 10px; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎯 TalentForce AI: The Professional Recruitment Suite")
st.markdown("---")

# --- SIDEBAR: RECRUITER CONFIGURATION ---
st.sidebar.header("📋 Job Requirements")
job_title = st.sidebar.text_input("Job Title", "e.g. Senior Marketing Executive")

# Inclusive Education Options
degree_options = ["Not Specified", "High School", "Diploma", "B.A.", "B.Com", "B.Sc", "B.Tech/B.E.", "BCA", "BBA", "M.A.", "M.Sc", "M.Tech", "MBA", "MCA", "PhD"]
req_degree = st.sidebar.selectbox("Minimum Education Required", degree_options)

# Experience Filter
min_exp = st.sidebar.slider("Minimum Years of Experience", 0, 15, 2)

# Professional Job Description
jd_input = st.sidebar.text_area("Detailed Job Description:", height=250, 
                                placeholder="Paste the full job post here including responsibilities and skills...")

# --- MAIN INTERFACE: CANDIDATE ANALYSIS ---
uploaded_file = st.file_uploader("📤 Upload Candidate Resume (PDF)", type="pdf")

if uploaded_file and jd_input:
    with st.spinner("Deep-scanning candidate profile..."):
        # 1. TEXT EXTRACTION
        reader = PyPDF2.PdfReader(uploaded_file)
        resume_text = " ".join([p.extract_text() for p in reader.pages])

        # 2. SEMANTIC ANALYSIS
        jd_doc = nlp(jd_input)
        res_doc = nlp(resume_text)
        match_score = round(jd_doc.similarity(res_doc) * 100, 2)

        # 3. YEARS OF EXPERIENCE EXTRACTION (Regex)
        exp_matches = re.findall(r'(\d+)\s*(?:years?|yrs?|yr)\s*(?:of)?\s*exp', resume_text.lower())
        years_found = int(exp_matches[0]) if exp_matches else 0
        
        # 4. EDUCATION VALIDATION
        edu_match = True if req_degree == "Not Specified" else req_degree.lower() in resume_text.lower()

        # 5. KEYWORD GAP ANALYSIS
        jd_keywords = set([t.text.lower() for t in jd_doc if t.pos_ in ["NOUN", "PROPN"] and not t.is_stop])
        res_keywords = set([t.text.lower() for t in res_doc if t.pos_ in ["NOUN", "PROPN"]])
        missing = list(jd_keywords - res_keywords)

        # --- UI DISPLAY: THE SCORECARD ---
        st.header(f"Candidate Analysis: {uploaded_file.name}")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("AI Compatibility", f"{match_score}%")
        with c2:
            st.metric("Exp. Detected", f"{years_found} Years")
        with c3:
            status = "✅ Met" if edu_match else "❌ Missing"
            st.metric("Education", status)

        st.divider()

        # --- DETAILED FEEDBACK SECTION ---
        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.subheader("🕵️ Recruiter's Audit")
            
            # Detailed Logic-based Advice
            if match_score > 70 and years_found >= min_exp and edu_match:
                st.success("**VERDICT: HIGH PRIORITY CANDIDATE**")
                st.write(f"This candidate is a strong match for the **{job_title}** role. Their professional vocabulary aligns perfectly with your requirements.")
            elif match_score > 50:
                st.warning("**VERDICT: POTENTIAL MATCH (Needs Review)**")
                st.write("The candidate has the right background but may lack specific technical keywords or the required seniority level.")
            else:
                st.error("**VERDICT: NOT RECOMMENDED**")
                st.write("Significant gaps found between the candidate's resume and the job description requirements.")

        with col_b:
            st.subheader("🛠️ Actionable Improvement Plan")
            st.write("To improve the selection probability, the candidate should:")
            if not edu_match:
                st.info(f"👉 Explicitly list the **{req_degree}** degree in the Education section.")
            if years_found < min_exp:
                st.info(f"👉 Highlight relevant projects to compensate for the **{min_exp - years_found} year(s)** experience gap.")
            if missing:
                st.write("**Include these missing keywords:**")
                st.caption(", ".join([m.capitalize() for m in missing[:15]]))

        # --- DATA VISUALIZATION ---
        st.markdown("### 📈 Competency Map")
        chart_data = pd.DataFrame({
            'Category': ['AI Match', 'Experience', 'Education'],
            'Score': [match_score, (years_found/min_exp)*100 if min_exp > 0 else 100, 100 if edu_match else 0]
        })
        st.bar_chart(data=chart_data, x='Category', y='Score')

else:
    st.info("👋 Welcome! Set your hiring rules in the sidebar and upload a resume to start the deep analysis.")

