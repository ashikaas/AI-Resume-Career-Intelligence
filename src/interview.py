ROLE_QUESTIONS = {
    "Data Analyst": [
        "Explain how you would clean a dataset containing missing values and duplicates.",
        "What is the difference between INNER JOIN and LEFT JOIN in SQL?",
        "How would you analyze a sudden drop in sales?",
        "What is A/B testing and when would you use it?",
        "How would you create a dashboard for business stakeholders?"
    ],

    "Business Analyst": [
        "How would you gather requirements from a business stakeholder?",
        "What is the difference between a business requirement and a functional requirement?",
        "How would you analyze a business process that is underperforming?",
        "How would you communicate analytical findings to a non-technical stakeholder?",
        "What KPIs would you use to measure business performance?"
    ],

    "Machine Learning Engineer": [
        "Explain the difference between supervised and unsupervised learning.",
        "What is overfitting and how can you prevent it?",
        "Explain the difference between classification and regression.",
        "What is feature engineering and why is it important?",
        "How would you evaluate a machine learning model?"
    ],

    "AI Engineer": [
        "What is the difference between Artificial Intelligence and Machine Learning?",
        "Explain how a machine learning model is trained.",
        "What is NLP and where is it used?",
        "What are embeddings and why are they useful?",
        "How would you deploy an AI model into an application?"
    ],

    "Software Engineer": [
        "Explain the four main principles of Object-Oriented Programming.",
        "What is the difference between a list and a tuple in Python?",
        "What is an API and how does a REST API work?",
        "What is Git and why is it used?",
        "How would you debug a program that is producing incorrect results?"
    ]
}


def generate_interview_questions(role, skills):

    questions = ROLE_QUESTIONS.get(
        role,
        ROLE_QUESTIONS["AI Engineer"]
    )

    # Add skill-specific questions
    skill_questions = []

    if "python" in [s.lower() for s in skills]:
        skill_questions.append(
            "Explain how you would use Python to preprocess a real-world dataset."
        )

    if "sql" in [s.lower() for s in skills]:
        skill_questions.append(
            "Write a SQL query to find the top 5 customers based on total purchase value."
        )

    if "machine learning" in [s.lower() for s in skills]:
        skill_questions.append(
            "Explain the complete machine learning workflow from data preprocessing to model evaluation."
        )

    if "power bi" in [s.lower() for s in skills]:
        skill_questions.append(
            "How would you design a Power BI dashboard for management?"
        )

    if "pandas" in [s.lower() for s in skills]:
        skill_questions.append(
            "How would you use Pandas to handle missing values and duplicate records?"
        )

    return questions + skill_questions


def evaluate_answer(answer):

    if not answer or len(answer.strip()) < 20:
        return 0, "Answer is too short. Try explaining your approach with an example."

    word_count = len(answer.split())

    if word_count >= 80:
        score = 9
    elif word_count >= 50:
        score = 8
    elif word_count >= 30:
        score = 7
    else:
        score = 6

    feedback = (
        "Good attempt. Try to structure your answer using "
        "concept → approach → example → result."
    )

    return score, feedback