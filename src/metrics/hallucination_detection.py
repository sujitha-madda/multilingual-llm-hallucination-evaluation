def detect_hallucination(
    similarity_score,
    threshold=0.6
):

    return similarity_score < threshold
