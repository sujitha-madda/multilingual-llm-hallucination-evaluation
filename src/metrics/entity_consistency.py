import re


def extract_keywords(text):

    if not isinstance(text, str):
        return set()

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z0-9\s]',
        '',
        text
    )

    words = text.split()

    stopwords = set([
        "the", "is", "a", "an",
        "in", "on", "at", "of",
        "and", "to", "for",
        "with", "as", "by"
    ])

    keywords = [
        w for w in words
        if w not in stopwords and len(w) > 3
    ]

    return set(keywords)


def entity_overlap(a, b):

    set_a = extract_keywords(a)
    set_b = extract_keywords(b)

    if len(set_a) == 0:
        return 0

    return len(set_a & set_b) / len(set_a)
