from numpy.random import binomial
from numpy import mean
from scipy.special import expit as logistic

from .endogenous import Summarization, EndogenousVar

students_intelligence = Summarization(
    target="students",
    sieve=lambda p, s: lambda x: (x, p) in s['relations']['registered'],
    mapper=lambda x, s: int((x, "high") in s['attributes']['intelligence']),
    reducer=lambda xs: 0 if not xs else mean(xs)
)

professors_skill = Summarization(
    target="professors",
    sieve=lambda p, s,: lambda x: (x, p) in s['relations']['teaches'],
    mapper=lambda x, s: int((x, "high") in s['attributes']['teaching_skills']),
    reducer=sum
)

course_diff = Summarization(
    target="courses",
    sieve=lambda p, s: lambda x: x == p,
    mapper=lambda x, s: int((x, "high") in s['attributes']['difficulty']),
    reducer=lambda xs: xs[0]
)

tutoring = EndogenousVar(
    reducer=lambda xs: "yes" if binomial(1, logistic(sum(xs))) else "no",
    summarizations={
        students_intelligence,
        professors_skill,
        course_diff
    }
)
