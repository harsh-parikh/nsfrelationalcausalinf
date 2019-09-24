import json


def load_dataset():
    """Load OpenReview dataset.

    You probably want `authors, papers, reviews, confs = load_dataset()`"""
    with open("../openreview-dataset/results/authors.json", "r") as f:
        authors = json.load(f)

    with open("../openreview-dataset/results/papers.json", "r") as f:
        papers = json.load(f)

    with open("../openreview-dataset/results/reviews.json", "r") as f:
        reviews = json.load(f)

    with open("../openreview-dataset/results/confs.json", "r") as f:
        confs = json.load(f)

    return (authors, papers, reviews, confs)


def prestigious(ranking):
    # "corp" means one of few prestigious corporations (e.g. MSR)
    if ranking == "corp":
        return True

    # either low-ranking corporation or university that is not in top 2000
    if ranking == "":
        return False

    # rankings past 100 are in the form "150-200".
    if "-" in ranking:
        return False

    return int(ranking) < 40
