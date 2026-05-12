import torch
import torch.nn.functional as F


def cosine_similarity(embedding1, embedding2):

    similarity = F.cosine_similarity(
        embedding1,
        embedding2
    )

    return similarity.item()


def classify_pair(similarity, threshold):

    if similarity >= threshold:
        return 1

    return 0