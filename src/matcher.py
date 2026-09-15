from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_text_similarity(resume_text, job_description):
    """
    Calculate similarity between resume and job description
    using TF-IDF and cosine similarity.
    """

    if not resume_text or not job_description:
        return 0.0

    documents = [
        resume_text,
        job_description
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    return round(float(similarity) * 100, 2)


def calculate_skill_match(resume_skills, job_skills):
    """
    Calculate percentage of required job skills
    found in the resume.
    """

    if not job_skills:
        return 0.0

    resume_set = set(resume_skills)
    job_set = set(job_skills)

    matched_skills = resume_set.intersection(job_set)

    score = (len(matched_skills) / len(job_set)) * 100

    return round(score, 2)


def get_skill_gap(resume_skills, job_skills):
    """
    Return matched and missing skills.
    """

    resume_set = set(resume_skills)
    job_set = set(job_skills)

    matched = sorted(resume_set.intersection(job_set))
    missing = sorted(job_set.difference(resume_set))

    return matched, missing


def calculate_final_score(skill_score, text_score):
    """
    Combine skill match and text similarity.

    Skill matching has a higher weight because
    required skills are important for job suitability.
    """

    final_score = (
        skill_score * 0.70
        + text_score * 0.30
    )

    return round(final_score, 2)