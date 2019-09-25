import json

# universities ranked higher than this are prestigious
PRESTIGE_CUTOFF = 40


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


def prestigious(author):
    ranking = author["world_rank"]

    # "corp" means one of few prestigious corporations (e.g. MSR)
    if ranking == "corp":
        return True

    # either low-ranking corporation or university that is not in top 2000
    if ranking == "":
        return False

    # rankings past 100 are in the form "150-200".
    if "-" in ranking:
        return False

    return int(ranking) < PRESTIGE_CUTOFF


def publishing_years(author):
    """Return range of publication years, given an author object."""
    try:
        pub_info = author["scopus"]["_json"]["author-profile"]["publication-range"]
        return int(pub_info["@end"]) - int(pub_info["@start"])
    except (KeyError, TypeError):
        return 0


def h_index(author):
    try:
        return int(author["scopus"]["_json"]["h-index"])
    except (KeyError, TypeError):
        return 0
