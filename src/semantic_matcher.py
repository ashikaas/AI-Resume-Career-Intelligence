from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load the pre-trained sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def calculate_semantic_similarity(resume_text, job_description):

    if not resume_text or not job_description:
        return 0.0

    # Convert resume and job description into numerical vectors
    embeddings = model.encode(
        [resume_text, job_description]
    )

    # Calculate similarity between the two vectors
    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return round(float(similarity) * 100, 2)