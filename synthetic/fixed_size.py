'''
    This script generates intermediate embedding tables
    to allow us to debug our learning models
'''
import torch
import numpy as np
from scipy.special import expit as logistic

import random

random.seed(344)

'''
    Parameters
'''
epochs = 200

# entities
courses = range(0, 5000)
instructors = range(0, 6000)
students = range(0, 25000)

# embedding sizes
class_size = 25
class_inst = 2

# probabilities
p_intelligent_student = 0.5
p_skilled_instructor = 0.5
p_difficult_course = 0.5

difficulty = np.array([random.random() < p_difficult_course
                       for _ in courses]).reshape(-1, 1)
skill = np.array([random.random() < p_skilled_instructor
                  for _ in instructors]).reshape(-1, 1)
intelligence = np.array([random.random() < p_intelligent_student
                  for _ in students]).reshape(-1, 1)

registered = []
lecturers = []

for c in courses:
    course_students = random.sample(students, class_size)
    registered.append([intelligence[s][0] for s in course_students])

    course_instructors = random.sample(instructors, class_inst)
    lecturers.append([skill[i][0] for i in course_instructors])

registered = np.array(registered)
lecturers = np.array(lecturers)

embeddings = []
p_labels = []
labels = []
for c in courses:
    e = [
        difficulty[c][0] - 0.5,
        np.mean(registered[c]) - 0.5,
        np.sum(lecturers[c] / 2) - 0.5
    ]
    embeddings.append(e)
    p = logistic(sum(e))
    p_labels.append([p])
    labels.append([random.random() < p])


embeddings = np.array(embeddings)
p_labels = np.array(p_labels)
labels = np.array(labels, dtype=np.uint8)

'''
  Now attempt to teach a classifier these embeddings
'''

class SimpleClassifier(torch.nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(3, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)


model = SimpleClassifier()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

labels_t = torch.FloatTensor(labels)
embeddings_t = torch.Tensor(embeddings)
p_labels_t = torch.FloatTensor(p_labels)

for i in range(epochs):
    optimizer.zero_grad()
    output = model(embeddings_t)
    loss = criterion(output, p_labels_t)
    loss.backward()
    optimizer.step()
    print(f"loss: {loss:.04}, " 
          f"{((p_labels_t - output) < 0.05).sum()} / 5000 correct")


print(f"weights: {model.model[0].weight.detach().numpy()}")
