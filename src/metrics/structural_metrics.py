def repetition(text):

    words = text.split()

    if len(words) == 0:
        return 0

    return 1 - (len(set(words)) / len(words))


def length_ratio(generated, reference):

    return len(generated.split()) / (
        len(reference.split()) + 1e-5
    )


def drift(similarity_function, question, generated):

    return similarity_function(
        question,
        generated
    )
