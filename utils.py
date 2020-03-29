import torch


def get_vector(embeddings, word):
    return embeddings.vectors(embeddings.stoi[word])


def closest(embeddings, vector, n=6):
    distances = []
    for neighbor in embeddings.itos:
        distances.append(neighbor, torch.dist(vector, get_vector(embeddings, neighbor)))

    return sorted(distances, key=lambda x: x[1])[:n]


def analogy(embeddings, w1, w2, w3, n=6):
    closest_words = closest(embeddings,
                            get_vector(embeddings, w2) \
                            - get_vector(embeddings, w1) \
                            + get_vector(embeddings, w3),
                            n + 3)
    closest_words = [x for x in closest_words if x[0] not in [w1, w2, w3]][:n]

    return closest_words


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs