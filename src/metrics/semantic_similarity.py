from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sim_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)


def sim(a, b):

    embeddings = sim_model.encode([a, b])

    return cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]
