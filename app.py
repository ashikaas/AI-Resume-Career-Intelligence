import streamlit as st
import pandas as pd
import plotly.express as px

from src.resume_parser import extract_text_from_pdf
from src.preprocessing import clean_text
from src.skill_extractor import extract_skills

from src.matcher import (
    calculate_text_similarity,
    calculate_skill_match,
    get_skill_gap,
    calculate_final_score
)

from src.recommender import generate_recommendations
from src.ml_model import load_model, predict_roles
from src.semantic_matcher import calculate_semantic_similarity

from src.interview import (
    generate_interview_questions,
    evaluate_answer
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Resume & Career Intelligence",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if "cleaned_resume" not in st.session_state:
    st.session_state.cleaned_resume = ""

if "resume_skills" not in st.session_state:
    st.session_state.resume_skills = []

if "job_skills" not in st.session_state:
    st.session_state.job_skills = []

if "matched_skills" not in st.session_state:
    st.session_state.matched_skills = []

if "missing_skills" not in st.session_state:
    st.session_state.missing_skills = []

if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

if "text_score" not in st.session_state:
    st.session_state.text_score = 0.0

if "semantic_score" not in st.session_state:
    st.session_state.semantic_score = 0.0

if "skill_score" not in st.session_state:
    st.session_state.skill_score = 0.0

if "final_score" not in st.session_state:
    st.session_state.final_score = 0.0

if "role_predictions" not in st.session_state:
    st.session_state.role_predictions = []

if "job_option" not in st.session_state:
    st.session_state.job_option = ""

if "job_description" not in st.session_state:
    st.session_state.job_description = ""

if "interview_role" not in st.session_state:
    st.session_state.interview_role = ""

if "answer_scores" not in st.session_state:
    st.session_state.answer_scores = {}

if "answer_feedback" not in st.session_state:
    st.session_state.answer_feedback = {}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.hero {
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    background: linear-gradient(
        135deg,
        rgba(80, 80, 200, 0.15),
        rgba(100, 200, 200, 0.10)
    );
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 18px;
}

.skill-pill {
    display: inline-block;
    padding: 7px 12px;
    margin: 4px;
    border-radius: 20px;
    background-color: rgba(80, 150, 220, 0.15);
    border: 1px solid rgba(80, 150, 220, 0.3);
}

.missing-pill {
    display: inline-block;
    padding: 7px 12px;
    margin: 4px;
    border-radius: 20px;
    background-color: rgba(220, 100, 100, 0.12);
    border: 1px solid rgba(220, 100, 100, 0.3);
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<h1>🤖 AI Resume & Career Intelligence</h1>

<p>
Analyze your resume, match it with jobs, identify skill gaps,
predict suitable career roles and practice for interviews.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🚀 AI Career Pipeline")

    st.markdown("""
    **1. Resume Extraction**  
    **2. NLP Preprocessing**  
    **3. Skill Extraction**  
    **4. TF-IDF Matching**  
    **5. Semantic Matching**  
    **6. Skill Gap Analysis**  
    **7. ML Role Prediction**  
    **8. Learning Recommendations**  
    **9. AI Mock Interview**  
    **10. Career Readiness**
    """)

    st.divider()

    st.subheader("Current Features")

    st.write("✅ PDF Resume Parsing")
    st.write("✅ NLP Preprocessing")
    st.write("✅ Skill Extraction")
    st.write("✅ TF-IDF Matching")
    st.write("✅ Semantic Matching")
    st.write("✅ Skill Gap Detection")
    st.write("✅ ML Role Prediction")
    st.write("✅ Learning Recommendations")
    st.write("✅ AI Mock Interview")
    st.write("✅ Answer Evaluation")

    st.divider()

    st.caption(
        "AI Resume & Career Intelligence"
    )


# ============================================================
# LOAD JOB DATA
# ============================================================

jobs_df = pd.read_csv(
    "data/jobs.csv"
)


# ============================================================
# RESUME UPLOAD
# ============================================================

st.subheader("📄 Upload Your Resume")

resume_file = st.file_uploader(
    "Upload your resume in PDF format",
    type=["pdf"],
    key="resume_uploader"
)

if resume_file:

    st.success(
        f"Resume uploaded: {resume_file.name}"
    )


# ============================================================
# JOB SELECTION
# ============================================================

st.subheader("💼 Select Target Job")

job_option = st.selectbox(
    "Choose a job role",
    jobs_df["job_title"].tolist()
    + ["Custom Job Description"],
    key="job_selector"
)


job_description = ""


if job_option == "Custom Job Description":

    job_description = st.text_area(
        "Paste the job description here",
        height=200,
        key="custom_job_description"
    )

else:

    selected_job = jobs_df[
        jobs_df["job_title"] == job_option
    ]

    if not selected_job.empty:

        job_description = selected_job.iloc[0][
            "job_description"
        ]


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_button = st.button(
    "🚀 Analyze Resume",
    type="primary",
    use_container_width=True
)


# ============================================================
# RUN ANALYSIS
# ============================================================

if analyze_button:

    if resume_file is None:

        st.warning(
            "Please upload a resume PDF first."
        )

    elif not job_description.strip():

        st.warning(
            "Please provide a job description."
        )

    else:

        with st.spinner(
            "Analyzing your resume..."
        ):

            # ------------------------------------------------
            # RESUME TEXT
            # ------------------------------------------------

            resume_text = extract_text_from_pdf(
                resume_file
            )

            # ------------------------------------------------
            # CLEAN TEXT
            # ------------------------------------------------

            cleaned_resume = clean_text(
                resume_text
            )

            cleaned_job = clean_text(
                job_description
            )

            # ------------------------------------------------
            # SKILLS
            # ------------------------------------------------

            resume_skills = extract_skills(
                cleaned_resume
            )

            job_skills = extract_skills(
                cleaned_job
            )

            # ------------------------------------------------
            # TF-IDF
            # ------------------------------------------------

            text_score = calculate_text_similarity(
                cleaned_resume,
                cleaned_job
            )

            # ------------------------------------------------
            # SEMANTIC SIMILARITY
            # ------------------------------------------------

            semantic_score = calculate_semantic_similarity(
                cleaned_resume,
                cleaned_job
            )

            # ------------------------------------------------
            # SKILL MATCH
            # ------------------------------------------------

            skill_score = calculate_skill_match(
                resume_skills,
                job_skills
            )

            # ------------------------------------------------
            # FINAL SCORE
            # ------------------------------------------------

            final_score = calculate_final_score(
                skill_score,
                text_score
            )

            # ------------------------------------------------
            # SKILL GAP
            # ------------------------------------------------

            matched_skills, missing_skills = get_skill_gap(
                resume_skills,
                job_skills
            )

            # ------------------------------------------------
            # RECOMMENDATIONS
            # ------------------------------------------------

            recommendations = generate_recommendations(
                missing_skills
            )

            # ------------------------------------------------
            # ML MODEL
            # ------------------------------------------------

            model = load_model()

            role_predictions = predict_roles(
                model,
                cleaned_resume
            )


            # =================================================
            # SAVE EVERYTHING TO SESSION STATE
            # =================================================

            st.session_state.analysis_done = True

            st.session_state.resume_text = resume_text

            st.session_state.cleaned_resume = cleaned_resume

            st.session_state.resume_skills = resume_skills

            st.session_state.job_skills = job_skills

            st.session_state.matched_skills = matched_skills

            st.session_state.missing_skills = missing_skills

            st.session_state.recommendations = recommendations

            st.session_state.text_score = text_score

            st.session_state.semantic_score = semantic_score

            st.session_state.skill_score = skill_score

            st.session_state.final_score = final_score

            st.session_state.role_predictions = role_predictions

            st.session_state.job_option = job_option

            st.session_state.job_description = job_description

            # Set default interview role

            if role_predictions:

                st.session_state.interview_role = (
                    role_predictions[0][0]
                )


# ============================================================
# SHOW RESULTS
# ============================================================

if st.session_state.analysis_done:

    # ========================================================
    # RESULTS HEADER
    # ========================================================

    st.divider()

    st.header(
        "📊 Resume Analysis Results"
    )


    # ========================================================
    # GET RESULTS FROM SESSION STATE
    # ========================================================

    resume_text = st.session_state.resume_text

    resume_skills = st.session_state.resume_skills

    job_skills = st.session_state.job_skills

    matched_skills = st.session_state.matched_skills

    missing_skills = st.session_state.missing_skills

    recommendations = st.session_state.recommendations

    text_score = st.session_state.text_score

    semantic_score = st.session_state.semantic_score

    skill_score = st.session_state.skill_score

    final_score = st.session_state.final_score

    role_predictions = st.session_state.role_predictions


    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Overall Match",
            f"{final_score:.1f}%"
        )

    with col2:

        st.metric(
            "Skill Match",
            f"{skill_score:.1f}%"
        )

    with col3:

        st.metric(
            "TF-IDF Similarity",
            f"{text_score:.1f}%"
        )

    with col4:

        st.metric(
            "Semantic Similarity",
            f"{semantic_score:.1f}%"
        )

    with col5:

        st.metric(
            "Skills Matched",
            len(matched_skills)
        )


    # ========================================================
    # CAREER READINESS
    # ========================================================

    st.subheader(
        "🎯 Career Readiness"
    )

    readiness_score = final_score

    st.progress(
        min(int(readiness_score), 100)
    )

    if readiness_score >= 80:

        st.success(
            f"Excellent readiness — "
            f"{readiness_score:.1f}%"
        )

    elif readiness_score >= 60:

        st.info(
            f"Good readiness — "
            f"{readiness_score:.1f}%"
        )

    elif readiness_score >= 40:

        st.warning(
            f"Moderate readiness — "
            f"{readiness_score:.1f}%"
        )

    else:

        st.error(
            f"Needs improvement — "
            f"{readiness_score:.1f}%"
        )


    # ========================================================
    # ML JOB ROLE PREDICTION
    # ========================================================

    st.divider()

    st.header(
        "🤖 AI Job Role Prediction"
    )

    st.write(
        "The machine-learning model analyzes your resume "
        "and predicts suitable career roles."
    )

    if role_predictions:

        top_role, top_probability = role_predictions[0]

        st.success(
            f"🎯 **Top Predicted Role: {top_role}** "
            f"({top_probability * 100:.2f}% probability)"
        )


        prediction_data = []

        for role, probability in role_predictions[:5]:

            prediction_data.append({
                "Job Role": role,
                "Probability": round(
                    probability * 100,
                    2
                )
            })


        prediction_df = pd.DataFrame(
            prediction_data
        )


        st.dataframe(
            prediction_df,
            use_container_width=True,
            hide_index=True
        )


        fig_roles = px.bar(
            prediction_df,
            x="Probability",
            y="Job Role",
            orientation="h",
            title="Predicted Job Roles"
        )

        fig_roles.update_layout(
            xaxis_title="Probability (%)",
            yaxis_title="Job Role"
        )

        st.plotly_chart(
            fig_roles,
            use_container_width=True
        )


    # ========================================================
    # SKILL ANALYSIS
    # ========================================================

    st.divider()

    st.header(
        "🧠 Skill Analysis"
    )


    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # MATCHED SKILLS
    # --------------------------------------------------------

    with col1:

        st.subheader(
            "✅ Matched Skills"
        )

        if matched_skills:

            for skill in matched_skills:

                st.markdown(
                    f'<span class="skill-pill">'
                    f'{skill}'
                    f'</span>',
                    unsafe_allow_html=True
                )

        else:

            st.write(
                "No matching skills found."
            )


    # --------------------------------------------------------
    # MISSING SKILLS
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "⚠️ Missing Skills"
        )

        if missing_skills:

            for skill in missing_skills:

                st.markdown(
                    f'<span class="missing-pill">'
                    f'{skill}'
                    f'</span>',
                    unsafe_allow_html=True
                )

        else:

            st.success(
                "No major skill gaps detected!"
            )


    # ========================================================
    # SKILL CHART
    # ========================================================

    st.subheader(
        "📈 Skill Match Overview"
    )

    chart_df = pd.DataFrame({
        "Category": [
            "Matched Skills",
            "Missing Skills"
        ],
        "Count": [
            len(matched_skills),
            len(missing_skills)
        ]
    })


    fig_skills = px.bar(
        chart_df,
        x="Category",
        y="Count",
        title="Skills Required vs Skills Available"
    )


    st.plotly_chart(
        fig_skills,
        use_container_width=True
    )


    # ========================================================
    # LEARNING PLAN
    # ========================================================

    st.divider()

    st.header(
        "📚 Personalized Learning Plan"
    )


    if recommendations:

        for index, recommendation in enumerate(
            recommendations,
            start=1
        ):

            st.markdown(
                f"**{index}.** {recommendation}"
            )

    else:

        st.success(
            "No additional learning recommendations required."
        )


    # ========================================================
    # DETECTED SKILLS
    # ========================================================

    st.divider()

    st.header(
        "🔎 Skills Detected in Your Resume"
    )


    if resume_skills:

        for skill in resume_skills:

            st.markdown(
                f'<span class="skill-pill">'
                f'{skill}'
                f'</span>',
                unsafe_allow_html=True
            )

    else:

        st.warning(
            "No predefined skills were detected."
        )


    # ========================================================
    # RESUME TEXT
    # ========================================================

    st.divider()

    with st.expander(
        "📄 View Extracted Resume Text"
    ):

        st.text_area(
            "Extracted Text",
            resume_text,
            height=300,
            key="extracted_resume_text"
        )


    # ========================================================
    # AI MOCK INTERVIEW
    # ========================================================

    st.divider()

    st.header(
        "🎤 AI Mock Interview"
    )

    st.write(
        "Practice questions based on your predicted "
        "career role and detected skills."
    )


    # --------------------------------------------------------
    # PREDICTED ROLES
    # --------------------------------------------------------

    predicted_roles = [
        role
        for role, probability
        in role_predictions
    ]


    if predicted_roles:

        # IMPORTANT:
        # Don't overwrite the selected role on every rerun.

        if (
            st.session_state.interview_role
            not in predicted_roles
        ):

            st.session_state.interview_role = (
                predicted_roles[0]
            )


        interview_role = st.selectbox(
            "Choose your interview role",
            predicted_roles,
            key="interview_role"
        )


        # ----------------------------------------------------
        # GENERATE QUESTIONS
        # ----------------------------------------------------

        interview_questions = generate_interview_questions(
            interview_role,
            resume_skills
        )


        st.subheader(
            "🧑‍💻 Interview Questions"
        )


        # ----------------------------------------------------
        # QUESTIONS
        # ----------------------------------------------------

        for index, question in enumerate(
            interview_questions[:8],
            start=1
        ):

            st.markdown(
                f"### Question {index}"
            )

            st.write(
                question
            )


            answer = st.text_area(
                "Your Answer",
                key=f"answer_{index}",
                height=130,
                placeholder=(
                    "Type your answer here..."
                )
            )


            # ------------------------------------------------
            # EVALUATE BUTTON
            # ------------------------------------------------

            if st.button(
                f"Evaluate Answer {index}",
                key=f"evaluate_{index}"
            ):

                score, feedback = evaluate_answer(
                    answer
                )

                st.session_state.answer_scores[
                    index
                ] = score

                st.session_state.answer_feedback[
                    index
                ] = feedback


            # ------------------------------------------------
            # SHOW STORED RESULT
            # ------------------------------------------------

            if index in st.session_state.answer_scores:

                st.metric(
                    "Answer Score",
                    f"{st.session_state.answer_scores[index]}/10"
                )

                st.info(
                    st.session_state.answer_feedback[index]
                )


        # ====================================================
        # OVERALL INTERVIEW SCORE
        # ====================================================

        if st.session_state.answer_scores:

            st.divider()

            st.subheader(
                "📊 Interview Performance"
            )

            scores = list(
                st.session_state.answer_scores.values()
            )

            overall_interview_score = (
                sum(scores) / len(scores)
            )

            st.metric(
                "Overall Interview Score",
                f"{overall_interview_score:.1f}/10"
            )


            if overall_interview_score >= 8:

                st.success(
                    "Excellent interview performance!"
                )

            elif overall_interview_score >= 6:

                st.info(
                    "Good performance. Keep practicing."
                )

            else:

                st.warning(
                    "You need more interview practice."
                )


        # ====================================================
        # INTERVIEW TIPS
        # ====================================================

        st.subheader(
            "💡 Interview Tips"
        )

        st.markdown("""
        - Explain the **concept first**.
        - Describe your **approach step-by-step**.
        - Give a **real project example**.
        - Mention the **tools or technologies** you used.
        - Explain the **result or business impact**.
        - For technical questions, explain **why** you chose your approach.
        """)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Resume & Career Intelligence System • "
    "Python • NLP • Machine Learning • "
    "Sentence Transformers • Streamlit"
)