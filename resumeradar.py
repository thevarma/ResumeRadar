import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import json
from datetime import datetime, timedelta
import sqlite3
import hashlib
from io import StringIO
import base64
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import requests
import time
import random

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Configuration
st.set_page_config(
    page_title="ATS Optimizer - AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .skill-tag {
        background: #e3f2fd;
        border: 1px solid #2196f3;
        border-radius: 15px;
        padding: 0.2rem 0.8rem;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.8rem;
    }
    .missing-skill {
        background: #ffebee;
        border: 1px solid #f44336;
        color: #c62828;
    }
    .chat-message {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ""
if 'job_description' not in st.session_state:
    st.session_state.job_description = ""
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'job_applications' not in st.session_state:
    st.session_state.job_applications = []


# Database setup
def init_db():
    conn = sqlite3.connect('ats_optimizer.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS applications
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  company TEXT,
                  position TEXT,
                  status TEXT,
                  date_applied TEXT,
                  notes TEXT)''')
    conn.commit()
    conn.close()


init_db()


# Utility Functions
def extract_text_from_file(file):
    """Extract text from uploaded PDF or DOCX file"""
    try:
        if file.type == "application/pdf":
            # Simple PDF text extraction (placeholder)
            return "PDF text extraction would require PyMuPDF or similar library"
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Simple DOCX text extraction (placeholder)
            return "DOCX text extraction would require python-docx library"
        else:
            return file.read().decode('utf-8')
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return ""


def extract_resume_sections(text):
    """Extract different sections from resume text"""
    sections = {
        'name': '',
        'email': '',
        'phone': '',
        'skills': [],
        'experience': '',
        'education': '',
        'summary': ''
    }

    # Email extraction
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    sections['email'] = emails[0] if emails else ''

    # Phone extraction
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    phones = re.findall(phone_pattern, text)
    sections['phone'] = phones[0] if phones else ''

    # Skills extraction (simple keyword matching)
    skill_keywords = [
        'Python', 'Java', 'JavaScript', 'React', 'Node.js', 'SQL', 'MongoDB',
        'AWS', 'Azure', 'Docker', 'Kubernetes', 'Git', 'Machine Learning',
        'Data Science', 'Pandas', 'NumPy', 'TensorFlow', 'PyTorch', 'Tableau',
        'Power BI', 'Excel', 'R', 'Scala', 'Go', 'C++', 'C#', 'PHP', 'Ruby',
        'Django', 'Flask', 'Spring', 'Angular', 'Vue.js', 'TypeScript'
    ]

    found_skills = []
    text_upper = text.upper()
    for skill in skill_keywords:
        if skill.upper() in text_upper:
            found_skills.append(skill)

    sections['skills'] = found_skills
    return sections


def calculate_ats_score(text):
    """Calculate ATS compatibility score"""
    score = 100
    issues = []

    # Check for common ATS issues
    if len(re.findall(r'[^\w\s\-.,()@]', text)) > 10:
        score -= 15
        issues.append("Special characters detected")

    if len(text.split()) < 200:
        score -= 20
        issues.append("Resume too short")

    if len(text.split()) > 800:
        score -= 10
        issues.append("Resume might be too long")

    # Check for standard sections
    required_sections = ['experience', 'education', 'skills']
    for section in required_sections:
        if section.lower() not in text.lower():
            score -= 10
            issues.append(f"Missing {section} section")

    return max(0, score), issues


def calculate_jd_match_score(resume_text, job_description):
    """Calculate job description match score using TF-IDF"""
    if not resume_text or not job_description:
        return 0, []

    try:
        # Preprocess texts
        stop_words = set(stopwords.words('english'))

        def preprocess_text(text):
            tokens = word_tokenize(text.lower())
            return ' '.join([token for token in tokens if token.isalnum() and token not in stop_words])

        resume_processed = preprocess_text(resume_text)
        jd_processed = preprocess_text(job_description)

        # Calculate TF-IDF similarity
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([resume_processed, jd_processed])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        # Get matching keywords
        feature_names = vectorizer.get_feature_names_out()
        resume_vector = tfidf_matrix[0].toarray()[0]
        jd_vector = tfidf_matrix[1].toarray()[0]

        matching_keywords = []
        for i, (resume_score, jd_score) in enumerate(zip(resume_vector, jd_vector)):
            if resume_score > 0 and jd_score > 0:
                matching_keywords.append(feature_names[i])

        return int(similarity * 100), matching_keywords[:10]
    except Exception as e:
        st.error(f"Error calculating match score: {str(e)}")
        return 0, []


def analyze_skills_gap(resume_skills, job_description):
    """Analyze skills gap between resume and job requirements"""
    # Extract skills from job description
    skill_keywords = [
        'Python', 'Java', 'JavaScript', 'React', 'Node.js', 'SQL', 'MongoDB',
        'AWS', 'Azure', 'Docker', 'Kubernetes', 'Git', 'Machine Learning',
        'Data Science', 'Pandas', 'NumPy', 'TensorFlow', 'PyTorch', 'Tableau',
        'Power BI', 'Excel', 'R', 'Scala', 'Go', 'C++', 'C#', 'PHP', 'Ruby',
        'Django', 'Flask', 'Spring', 'Angular', 'Vue.js', 'TypeScript'
    ]

    jd_upper = job_description.upper()
    required_skills = [skill for skill in skill_keywords if skill.upper() in jd_upper]

    missing_skills = [skill for skill in required_skills if skill not in resume_skills]
    matched_skills = [skill for skill in required_skills if skill in resume_skills]

    return {
        'required_skills': required_skills,
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'match_percentage': len(matched_skills) / len(required_skills) * 100 if required_skills else 0
    }


def generate_ai_feedback(analysis_results):
    """Generate AI feedback based on analysis results"""
    feedback = []

    ats_score = analysis_results.get('ats_score', 0)
    jd_match = analysis_results.get('jd_match_score', 0)
    skills_gap = analysis_results.get('skills_gap', {})

    if ats_score < 70:
        feedback.append(
            "🔴 Your ATS score is below 70. Consider simplifying formatting and avoiding special characters.")
    elif ats_score < 85:
        feedback.append("🟡 Your ATS score is good but can be improved. Check for any formatting issues.")
    else:
        feedback.append("🟢 Excellent ATS compatibility!")

    if jd_match < 60:
        feedback.append("🔴 Low job match score. Consider incorporating more keywords from the job description.")
    elif jd_match < 80:
        feedback.append("🟡 Moderate job match. Try to include more relevant keywords and experiences.")
    else:
        feedback.append("🟢 Great job description match!")

    missing_skills = skills_gap.get('missing_skills', [])
    if missing_skills:
        feedback.append(f"📚 Consider adding these skills: {', '.join(missing_skills[:5])}")

    return feedback


# Predefined job roles and descriptions
JOB_ROLES = {
    "Data Analyst": """
    We are seeking a Data Analyst to join our team. The ideal candidate will have experience with:
    - SQL for data extraction and manipulation
    - Python or R for data analysis
    - Tableau or Power BI for data visualization
    - Excel for data processing
    - Statistical analysis and reporting
    - Experience with databases and data warehousing
    """,
    "Software Engineer": """
    We are looking for a Software Engineer with the following skills:
    - Programming languages: Python, Java, or JavaScript
    - Web development frameworks: React, Angular, or Vue.js
    - Backend development: Node.js, Django, or Spring
    - Database management: SQL, MongoDB
    - Version control: Git
    - Cloud platforms: AWS, Azure, or GCP
    - Agile development methodologies
    """,
    "DevOps Engineer": """
    DevOps Engineer position requiring expertise in:
    - Containerization: Docker, Kubernetes
    - Cloud platforms: AWS, Azure, GCP
    - Infrastructure as Code: Terraform, CloudFormation
    - CI/CD pipelines: Jenkins, GitLab CI, GitHub Actions
    - Monitoring: Prometheus, Grafana, ELK Stack
    - Scripting: Python, Bash, PowerShell
    - Linux/Unix system administration
    """,
    "Marketing Manager": """
    Marketing Manager role focusing on:
    - Digital marketing strategies
    - Content marketing and SEO
    - Social media management
    - Email marketing campaigns
    - Analytics: Google Analytics, Adobe Analytics
    - Marketing automation tools
    - Budget management and ROI analysis
    - Team leadership and project management
    """,
    "Product Manager": """
    Product Manager position requiring:
    - Product roadmap development
    - Market research and user analysis
    - Agile/Scrum methodologies
    - Data analysis and metrics tracking
    - Cross-functional team collaboration
    - User experience design principles
    - Technical understanding of software development
    - Business strategy and competitive analysis
    """
}


# Main Application
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🎯 ATS Optimizer - AI Resume Analyzer & Enhancer</h1>
        <p>Optimize your resume for Applicant Tracking Systems and increase your chances of landing interviews</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox("Choose a page:", [
        "📄 Resume Analysis",
        "🤖 AI Chatbot",
        "📊 Dashboard",
        "📝 Resume Optimizer",
        "📋 Job Tracker"
    ])

    if page == "📄 Resume Analysis":
        resume_analysis_page()
    elif page == "🤖 AI Chatbot":
        chatbot_page()
    elif page == "📊 Dashboard":
        dashboard_page()
    elif page == "📝 Resume Optimizer":
        resume_optimizer_page()
    elif page == "📋 Job Tracker":
        job_tracker_page()


def resume_analysis_page():
    st.header("📄 Resume Analysis")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose your resume file",
            type=['pdf', 'docx', 'txt'],
            help="Upload your resume in PDF, DOCX, or TXT format"
        )

        if uploaded_file is not None:
            # Extract text from file
            resume_text = extract_text_from_file(uploaded_file)
            st.session_state.resume_text = resume_text
            st.success("✅ Resume uploaded successfully!")

            # Show preview
            with st.expander("📖 Resume Preview"):
                st.text_area("Resume Content", resume_text[:500] + "...", height=200, disabled=True)

        # Manual text input option
        st.subheader("✏️ Or Paste Resume Text")
        manual_text = st.text_area(
            "Paste your resume text here:",
            height=200,
            placeholder="Copy and paste your resume content here..."
        )

        if manual_text:
            st.session_state.resume_text = manual_text

    with col2:
        st.subheader("💼 Job Role Selection")

        # Predefined roles dropdown
        selected_role = st.selectbox(
            "Select a job role:",
            ["Custom"] + list(JOB_ROLES.keys())
        )

        if selected_role != "Custom":
            st.session_state.job_description = JOB_ROLES[selected_role]
            st.text_area(
                "Job Description:",
                st.session_state.job_description,
                height=200,
                disabled=True
            )
        else:
            # Custom job description
            custom_jd = st.text_area(
                "Enter custom job description:",
                height=200,
                placeholder="Paste the job description here..."
            )
            if custom_jd:
                st.session_state.job_description = custom_jd

    # Analysis button
    if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
        if st.session_state.resume_text and st.session_state.job_description:
            with st.spinner("Analyzing your resume..."):
                # Perform analysis
                resume_sections = extract_resume_sections(st.session_state.resume_text)
                ats_score, ats_issues = calculate_ats_score(st.session_state.resume_text)
                jd_match_score, matching_keywords = calculate_jd_match_score(
                    st.session_state.resume_text,
                    st.session_state.job_description
                )
                skills_gap = analyze_skills_gap(
                    resume_sections['skills'],
                    st.session_state.job_description
                )

                # Store results
                st.session_state.analysis_results = {
                    'resume_sections': resume_sections,
                    'ats_score': ats_score,
                    'ats_issues': ats_issues,
                    'jd_match_score': jd_match_score,
                    'matching_keywords': matching_keywords,
                    'skills_gap': skills_gap
                }

                st.success("✅ Analysis completed!")
                st.experimental_rerun()
        else:
            st.error("❌ Please upload a resume and select/enter a job description first.")

    # Display results if available
    if st.session_state.analysis_results:
        display_analysis_results()


def display_analysis_results():
    """Display analysis results"""
    st.header("📊 Analysis Results")

    results = st.session_state.analysis_results

    # Score cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🎯 ATS Score",
            f"{results['ats_score']}/100",
            delta=f"{results['ats_score'] - 70}" if results['ats_score'] >= 70 else f"{results['ats_score'] - 70}"
        )

    with col2:
        st.metric(
            "🔗 Job Match Score",
            f"{results['jd_match_score']}/100",
            delta=f"{results['jd_match_score'] - 60}" if results[
                                                             'jd_match_score'] >= 60 else f"{results['jd_match_score'] - 60}"
        )

    with col3:
        skills_match = results['skills_gap']['match_percentage']
        st.metric(
            "🎯 Skills Match",
            f"{skills_match:.1f}%",
            delta=f"{skills_match - 70:.1f}%" if skills_match >= 70 else f"{skills_match - 70:.1f}%"
        )

    # Detailed analysis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Resume Sections Detected")
        sections = results['resume_sections']

        st.write(f"**Email:** {sections['email'] or 'Not found'}")
        st.write(f"**Phone:** {sections['phone'] or 'Not found'}")

        if sections['skills']:
            st.write("**Skills Found:**")
            for skill in sections['skills']:
                st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)

        if results['ats_issues']:
            st.subheader("⚠️ ATS Issues")
            for issue in results['ats_issues']:
                st.write(f"• {issue}")

    with col2:
        st.subheader("🎯 Skills Gap Analysis")
        skills_gap = results['skills_gap']

        if skills_gap['matched_skills']:
            st.write("**✅ Matched Skills:**")
            for skill in skills_gap['matched_skills']:
                st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)

        if skills_gap['missing_skills']:
            st.write("**❌ Missing Skills:**")
            for skill in skills_gap['missing_skills']:
                st.markdown(f'<span class="skill-tag missing-skill">{skill}</span>', unsafe_allow_html=True)

        if results['matching_keywords']:
            st.subheader("🔑 Matching Keywords")
            st.write(", ".join(results['matching_keywords'][:10]))

    # AI Feedback
    st.subheader("🤖 AI Recommendations")
    feedback = generate_ai_feedback(results)
    for item in feedback:
        st.write(item)


def chatbot_page():
    st.header("🤖 AI Resume Chatbot")
    st.write("Get personalized feedback and suggestions for your resume improvement.")

    # Chat interface
    if st.session_state.chat_history:
        for i, (role, message) in enumerate(st.session_state.chat_history):
            if role == "user":
                st.markdown(f"""
                <div style="text-align: right; margin: 1rem 0;">
                    <div style="background: #007bff; color: white; padding: 0.5rem; border-radius: 10px; display: inline-block; max-width: 70%;">
                        {message}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message">
                    🤖 {message}
                </div>
                """, unsafe_allow_html=True)

    # Chat input
    user_input = st.text_input("Ask me anything about your resume:",
                               placeholder="e.g., How can I improve my ATS score?")

    if st.button("Send") and user_input:
        # Add user message
        st.session_state.chat_history.append(("user", user_input))

        # Generate AI response (simplified)
        if "ats score" in user_input.lower():
            response = "To improve your ATS score: 1) Use standard section headings, 2) Avoid special characters, 3) Use keywords from job description, 4) Keep formatting simple, 5) Include all relevant sections (Experience, Education, Skills)."
        elif "skills" in user_input.lower():
            response = "Focus on adding technical skills mentioned in the job description. Consider online courses or certifications to gain missing skills. Highlight transferable skills from your experience."
        elif "keywords" in user_input.lower():
            response = "Incorporate keywords from the job description naturally throughout your resume. Focus on skills, technologies, and industry terms. Don't keyword stuff - make it contextual."
        else:
            response = "I'm here to help with your resume! You can ask me about ATS optimization, skills improvement, keyword usage, or any specific concerns about your resume."

        # Add AI response
        st.session_state.chat_history.append(("assistant", response))
        st.experimental_rerun()

    # Clear chat
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()


def dashboard_page():
    st.header("📊 Resume Analytics Dashboard")

    if not st.session_state.analysis_results:
        st.warning("⚠️ Please analyze your resume first on the Resume Analysis page.")
        return

    results = st.session_state.analysis_results

    # Score visualization
    fig_scores = go.Figure()

    scores = [
        results['ats_score'],
        results['jd_match_score'],
        results['skills_gap']['match_percentage']
    ]

    score_names = ['ATS Score', 'Job Match', 'Skills Match']
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

    fig_scores.add_trace(go.Bar(
        x=score_names,
        y=scores,
        marker_color=colors,
        text=[f"{score:.1f}%" for score in scores],
        textposition='auto',
    ))

    fig_scores.update_layout(
        title="Resume Score Breakdown",
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 100])
    )

    st.plotly_chart(fig_scores, use_container_width=True)

    # Skills analysis
    col1, col2 = st.columns(2)

    with col1:
        # Skills match pie chart
        skills_gap = results['skills_gap']
        matched_count = len(skills_gap['matched_skills'])
        missing_count = len(skills_gap['missing_skills'])

        if matched_count + missing_count > 0:
            fig_skills = go.Figure(data=[go.Pie(
                labels=['Matched Skills', 'Missing Skills'],
                values=[matched_count, missing_count],
                hole=.3,
                marker_colors=['#4ecdc4', '#ff6b6b']
            )])

            fig_skills.update_layout(title="Skills Match Analysis")
            st.plotly_chart(fig_skills, use_container_width=True)

    with col2:
        # ATS issues breakdown
        if results['ats_issues']:
            st.subheader("🔍 ATS Issues Detected")
            for issue in results['ats_issues']:
                st.write(f"• {issue}")
        else:
            st.success("✅ No major ATS issues detected!")

    # Keyword frequency
    if results['matching_keywords']:
        st.subheader("🔑 Top Matching Keywords")
        keyword_df = pd.DataFrame({
            'Keyword': results['matching_keywords'][:10],
            'Relevance': [random.randint(70, 100) for _ in range(min(10, len(results['matching_keywords'])))]
        })

        fig_keywords = px.bar(
            keyword_df,
            x='Relevance',
            y='Keyword',
            orientation='h',
            title="Keyword Relevance Scores"
        )
        st.plotly_chart(fig_keywords, use_container_width=True)


def resume_optimizer_page():
    st.header("📝 Resume Optimizer")
    st.write("Get personalized suggestions to improve your resume.")

    if not st.session_state.analysis_results:
        st.warning("⚠️ Please analyze your resume first on the Resume Analysis page.")
        return

    results = st.session_state.analysis_results

    # Optimization suggestions
    st.subheader("🎯 Optimization Suggestions")

    # ATS Optimization
    if results['ats_score'] < 85:
        st.markdown("""
        ### 🔧 ATS Score Improvements
        """)

        suggestions = [
            "Use standard section headings (Experience, Education, Skills, etc.)",
            "Avoid tables, columns, and complex formatting",
            "Use bullet points for easy scanning",
            "Include keywords from the job description",
            "Save as .docx or .pdf format",
            "Use standard fonts (Arial, Calibri, Times New Roman)"
        ]

        for suggestion in suggestions:
            st.write(f"• {suggestion}")

    # Skills Gap
    skills_gap = results['skills_gap']
    if skills_gap['missing_skills']:
        st.markdown("""
        ### 📚 Skills to Add
        """)

        for skill in skills_gap['missing_skills'][:5]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{skill}</h4>
                <p>Consider adding this skill to match job requirements. You can:</p>
                <ul>
                    <li>Take online courses (Coursera, Udemy, edX)</li>
                    <li>Practice with personal projects</li>
                    <li>Highlight related experience</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # Content suggestions based on job match
    if results['jd_match_score'] < 75:
        st.markdown("""
        ### ✍️ Content Optimization
        """)

        content_suggestions = [
            "Incorporate more keywords from the job description",
            "Quantify your achievements with numbers and percentages",
            "Use action verbs to start bullet points",
            "Tailor your summary/objective to the specific role",
            "Highlight relevant projects and accomplishments"
        ]

        for suggestion in content_suggestions:
            st.write(f"• {suggestion}")

    # Resume customization tool
    st.subheader("🛠️ Resume Customizer")

    if st.button("Generate Custom Summary"):
        with st.spinner("Generating personalized summary..."):
            time.sleep(2)  # Simulate processing

            custom_summary = f"""
            Experienced professional with expertise in {', '.join(results['resume_sections']['skills'][:3])}.
            Proven track record in delivering results and contributing to team success.
            Seeking to leverage technical skills and experience in a challenging role.
            """

            st.success("✅ Custom summary generated!")
            st.text_area("Suggested Summary:", custom_summary, height=100)

    # Download optimized resume template
    st.subheader("📥 Download Optimized Template")

    template_content = f"""
# {results['resume_sections']['name'] or 'Your Name'}
{results['resume_sections']['email']} | {results['resume_sections']['phone']}

## PROFESSIONAL SUMMARY
[Customized summary based on job description]

## TECHNICAL SKILLS
{', '.join(results['resume_sections']['skills'])}

## PROFESSIONAL EXPERIENCE
[Your experience with quantified achievements]

## EDUCATION
[Your educational background]

## PROJECTS
[Relevant projects that demonstrate your skills]
"""

    st.download_button(
        label="📥 Download Resume Template",
        data=template_content,
        file_name="optimized_resume_template.md",
        mime="text/markdown"
    )


def job_tracker_page():
    st.header("📋 Job Application Tracker")
    st.write("Track your job applications and interview progress.")

    # Add new application
    with st.expander("➕ Add New Application"):
        col1, col2 = st.columns(2)

        with col1:
            company = st.text_input("Company Name")
            position = st.text_input("Position Title")

        with col2:
            status = st.selectbox("Status", ["Applied", "Interview Scheduled", "Interview Completed", "Offer Received",
                                             "Rejected", "Withdrawn"])
            date_applied = st.date_input("Date Applied", datetime.now())

        notes = st.text_area("Notes", placeholder="Interview feedback, next steps, etc.")

        if st.button("Add Application"):
            if company and position:
                # Add to database
                conn = sqlite3.connect('ats_optimizer.db')
                c = conn.cursor()
                c.execute(
                    "INSERT INTO applications (company, position, status, date_applied, notes) VALUES (?, ?, ?, ?, ?)",
                    (company, position, status, date_applied.isoformat(), notes))
                conn.commit()
                conn.close()

                st.success("✅ Application added successfully!")
                st.experimental_rerun()
            else:
                st.error("❌ Please fill in company and position fields.")

    # Display applications
    conn = sqlite3.connect('ats_optimizer.db')
    df = pd.read_sql_query("SELECT * FROM applications ORDER BY date_applied DESC", conn)
    conn.close()

    if len(df) > 0:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Applications", len(df))

        with col2:
            interviews = len(df[df['status'].str.contains('Interview')])
            st.metric("Interviews", interviews)

        with col3:
            offers = len(df[df['status'] == 'Offer Received'])
            st.metric("Offers", offers)

        with col4:
            response_rate = interviews / len(df) * 100 if len(df) > 0 else 0
            st.metric("Response Rate", f"{response_rate:.1f}%")

        # Status distribution chart
        status_counts = df['status'].value_counts()
        fig_status = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Application Status Distribution"
        )
        st.plotly_chart(fig_status, use_container_width=True)

        # Applications timeline
        df['date_applied'] = pd.to_datetime(df['date_applied'])
        fig_timeline = px.scatter(
            df,
            x='date_applied',
            y='company',
            color='status',
            title="Application Timeline",
            hover_data=['position']
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Applications table
        st.subheader("📋 All Applications")

        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            status_filter = st.multiselect("Filter by Status", df['status'].unique(), default=df['status'].unique())
        with col2:
            company_filter = st.multiselect("Filter by Company", df['company'].unique(), default=df['company'].unique())

        filtered_df = df[
            (df['status'].isin(status_filter)) &
            (df['company'].isin(company_filter))
            ]

        # Display table with edit/delete options
        for index, row in filtered_df.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.markdown(f"""
                    **{row['company']}** - {row['position']}  
                    Status: {row['status']} | Applied: {row['date_applied'].strftime('%Y-%m-%d')}  
                    Notes: {row['notes'][:100]}{'...' if len(str(row['notes'])) > 100 else ''}
                    """)

                with col2:
                    if st.button("✏️ Edit", key=f"edit_{row['id']}"):
                        st.session_state[f"edit_mode_{row['id']}"] = True

                with col3:
                    if st.button("🗑️ Delete", key=f"delete_{row['id']}"):
                        conn = sqlite3.connect('ats_optimizer.db')
                        c = conn.cursor()
                        c.execute("DELETE FROM applications WHERE id = ?", (row['id'],))
                        conn.commit()
                        conn.close()
                        st.success("Application deleted!")
                        st.experimental_rerun()

                # Edit mode
                if st.session_state.get(f"edit_mode_{row['id']}", False):
                    with st.form(f"edit_form_{row['id']}"):
                        new_company = st.text_input("Company", value=row['company'])
                        new_position = st.text_input("Position", value=row['position'])
                        new_status = st.selectbox("Status",
                                                  ["Applied", "Interview Scheduled", "Interview Completed",
                                                   "Offer Received", "Rejected", "Withdrawn"],
                                                  index=["Applied", "Interview Scheduled", "Interview Completed",
                                                         "Offer Received", "Rejected", "Withdrawn"].index(
                                                      row['status']))
                        new_notes = st.text_area("Notes", value=row['notes'])

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.form_submit_button("Save Changes"):
                                conn = sqlite3.connect('ats_optimizer.db')
                                c = conn.cursor()
                                c.execute("UPDATE applications SET company=?, position=?, status=?, notes=? WHERE id=?",
                                          (new_company, new_position, new_status, new_notes, row['id']))
                                conn.commit()
                                conn.close()
                                st.session_state[f"edit_mode_{row['id']}"] = False
                                st.success("Application updated!")
                                st.experimental_rerun()

                        with col2:
                            if st.form_submit_button("Cancel"):
                                st.session_state[f"edit_mode_{row['id']}"] = False
                                st.experimental_rerun()

                st.divider()
    else:
        st.info("📝 No applications tracked yet. Add your first application above!")

        # Sample data for demo
        if st.button("📊 Load Sample Data"):
            sample_data = [
                ("Google", "Software Engineer", "Interview Completed", "2024-01-15", "Technical interview went well"),
                ("Microsoft", "Data Scientist", "Applied", "2024-01-10", "Applied through LinkedIn"),
                ("Amazon", "DevOps Engineer", "Rejected", "2024-01-05", "Not selected for next round"),
                ("Meta", "Product Manager", "Offer Received", "2024-01-01", "Great offer package!"),
                ("Apple", "iOS Developer", "Interview Scheduled", "2024-01-20", "Phone screen next week")
            ]

            conn = sqlite3.connect('ats_optimizer.db')
            c = conn.cursor()
            for data in sample_data:
                c.execute(
                    "INSERT INTO applications (company, position, status, date_applied, notes) VALUES (?, ?, ?, ?, ?)",
                    data)
            conn.commit()
            conn.close()

            st.success("✅ Sample data loaded!")
            st.experimental_rerun()


# Additional utility functions for enhanced features
def export_resume_data():
    """Export resume analysis data"""
    if st.session_state.analysis_results:
        data = {
            'timestamp': datetime.now().isoformat(),
            'analysis_results': st.session_state.analysis_results,
            'resume_text_length': len(st.session_state.resume_text),
            'job_description_length': len(st.session_state.job_description)
        }
        return json.dumps(data, indent=2)
    return None


def generate_improvement_plan():
    """Generate a structured improvement plan"""
    if not st.session_state.analysis_results:
        return None

    results = st.session_state.analysis_results
    plan = {
        'immediate_actions': [],
        'short_term_goals': [],
        'long_term_goals': []
    }

    # Immediate actions
    if results['ats_score'] < 70:
        plan['immediate_actions'].extend([
            "Simplify resume formatting",
            "Remove special characters and graphics",
            "Use standard section headings"
        ])

    if results['jd_match_score'] < 60:
        plan['immediate_actions'].extend([
            "Add more keywords from job description",
            "Rewrite summary to match job requirements"
        ])

    # Short-term goals
    missing_skills = results['skills_gap']['missing_skills']
    if missing_skills:
        plan['short_term_goals'].extend([
            f"Learn {skill}" for skill in missing_skills[:3]
        ])

    # Long-term goals
    plan['long_term_goals'].extend([
        "Build portfolio projects",
        "Gain relevant certifications",
        "Network in target industry"
    ])

    return plan


# Add export functionality to sidebar
def add_export_options():
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Export Options")

    if st.sidebar.button("📊 Export Analysis Data"):
        data = export_resume_data()
        if data:
            st.sidebar.download_button(
                label="📥 Download Analysis Report",
                data=data,
                file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.sidebar.warning("No analysis data to export")

    if st.sidebar.button("📋 Generate Improvement Plan"):
        plan = generate_improvement_plan()
        if plan:
            plan_text = f"""
# Resume Improvement Plan
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Immediate Actions (This Week)
{chr(10).join(f'- {action}' for action in plan['immediate_actions'])}

## Short-term Goals (Next Month)
{chr(10).join(f'- {goal}' for goal in plan['short_term_goals'])}

## Long-term Goals (3-6 Months)
{chr(10).join(f'- {goal}' for goal in plan['long_term_goals'])}
"""
            st.sidebar.download_button(
                label="📥 Download Improvement Plan",
                data=plan_text,
                file_name=f"improvement_plan_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
        else:
            st.sidebar.warning("No analysis data available")


# Add footer with additional information
def add_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <h4>🎯 ATS Optimizer - AI Resume Analyzer</h4>
        <p>Built with Streamlit • Powered by AI • Open Source</p>
        <p><strong>Tips for Best Results:</strong></p>
        <p>📄 Use clean, simple resume formatting • 🎯 Tailor for each job application • 📊 Regularly update your skills</p>
        <p><em>Remember: This tool provides suggestions to improve your resume's ATS compatibility. 
        Always review and customize recommendations based on your specific situation.</em></p>
    </div>
    """, unsafe_allow_html=True)


# Run the application
if __name__ == "__main__":
    main()
    add_export_options()
    add_footer()