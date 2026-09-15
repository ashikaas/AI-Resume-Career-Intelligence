RECOMMENDATION_MAP = {
    "python": "Improve Python programming, especially functions, OOP, file handling and data processing.",
    "sql": "Practice SQL queries including JOINs, GROUP BY, subqueries, CTEs and window functions.",
    "power bi": "Learn Power BI dashboards, data modeling, DAX and interactive visualizations.",
    "tableau": "Learn Tableau for data visualization, dashboards and business reporting.",
    "excel": "Improve Excel skills including formulas, pivot tables, charts and data analysis.",
    "statistics": "Study descriptive statistics, probability, hypothesis testing and correlation.",
    "machine learning": "Learn supervised and unsupervised machine learning algorithms and model evaluation.",
    "deep learning": "Learn neural networks, backpropagation and deep learning frameworks.",
    "natural language processing": "Learn NLP concepts such as tokenization, embeddings, text classification and similarity.",
    "pandas": "Practice Pandas for data cleaning, transformation, filtering and analysis.",
    "numpy": "Practice NumPy arrays, indexing, mathematical operations and numerical computing.",
    "scikit-learn": "Practice building and evaluating machine learning models using Scikit-learn.",
    "tensorflow": "Learn TensorFlow fundamentals and neural network model development.",
    "pytorch": "Learn PyTorch fundamentals, tensors, neural networks and model training.",
    "git": "Learn Git commands, branching, commits, merging and collaborative workflows.",
    "postgresql": "Practice PostgreSQL including database design, joins, indexing and advanced SQL.",
    "mysql": "Practice MySQL queries, database design, joins and database management.",
    "fastapi": "Learn FastAPI for building Python-based REST APIs.",
    "flask": "Learn Flask for developing Python web applications and APIs.",
    "docker": "Learn Docker containers, images, Dockerfiles and basic deployment.",
    "aws": "Learn AWS fundamentals including compute, storage and cloud deployment.",
    "azure": "Learn Microsoft Azure fundamentals and cloud services.",
}


def generate_recommendations(missing_skills):
    recommendations = []

    for skill in missing_skills:
        skill_lower = skill.lower()

        if skill_lower in RECOMMENDATION_MAP:
            recommendations.append(RECOMMENDATION_MAP[skill_lower])
        else:
            recommendations.append(
                f"Develop your knowledge of {skill.title()} through practical projects and exercises."
            )

    return recommendations