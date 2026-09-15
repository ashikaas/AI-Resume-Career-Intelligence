import re


SKILLS = [
    # Programming
    "python",
    "java",
    "c++",
    "c#",
    "javascript",
    "typescript",

    # Data
    "sql",
    "excel",
    "power bi",
    "tableau",
    "pandas",
    "numpy",
    "matplotlib",
    "plotly",

    # Machine Learning
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "natural language processing",
    "nlp",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "keras",

    # Statistics
    "statistics",
    "a/b testing",
    "hypothesis testing",
    "regression",

    # Data Engineering
    "etl",
    "data engineering",
    "data warehouse",
    "spark",
    "hadoop",

    # Cloud
    "aws",
    "azure",
    "google cloud",
    "gcp",

    # Databases
    "postgresql",
    "mysql",
    "mongodb",

    # Software
    "git",
    "github",
    "rest api",
    "api",
    "fastapi",
    "flask",
    "django",

    # Other
    "streamlit",
    "docker",
    "linux"
]


def extract_skills(text):
    """
    Extract known technical skills from text.
    """

    if not text:
        return []

    text = text.lower()

    found_skills = []

    for skill in SKILLS:

        # Escape special characters in skill names
        pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"

        if re.search(pattern, text):
            found_skills.append(skill)

    return sorted(found_skills)