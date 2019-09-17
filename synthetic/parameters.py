'''
    Model Parameters
'''

# Number of professors
N_p = 50
# Number of students
N_s = 1000
# Number of courses
N_c = 50

# Probability a course is difficult
P_d = 0.25
# Probability a student has high aptitude
P_a  = 0.65
# Probability two students are friends
P_f = 1/100
# Probability student is registered for any course
P_r = 1/10
# Probability a student is female
P_female = 1/2

# Probability course falls under a certain department
dept = {"math": 0.5, "cs": 0.2, "social sciences": 0.2, "other": 0.1}

# Number of timesteps to use for the Gibbs Sampler
# TODO change this once more data
time = 250
