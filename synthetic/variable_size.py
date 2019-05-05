import numpy as np
import random
from sklearn import preprocessing
import torch
import torch.nn as nn
from scipy.special import expit as logistic
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

random.seed(344)

def pad(ls : list) -> list:
    result = []
    columns = max([len(l) for l in ls])

    for l in ls:
        result.append(l + [0] * (columns - len(l)))

    return result

'''
    Parameters
'''
epochs = 200

# entities
courses = range(0, 5000)
instructors = range(0, 6000)
students = range(0, 25000)

# embedding sizes
class_size = range(10, 50)
class_inst = range(1, 4)

# probabilities
p_intelligent_student = 0.5
p_skilled_instructor = 0.5
p_difficult_course = 0.5

difficulty = np.array([random.random() < p_difficult_course
                       for _ in courses], dtype=np.uint8).reshape(-1, 1)
skill = np.array([random.random() < p_skilled_instructor
                  for _ in instructors], dtype=np.uint8).reshape(-1, 1)
intelligence = np.array([random.random() < p_intelligent_student
                  for _ in students], dtype=np.uint8).reshape(-1, 1)

registered = []
lecturers = []

for c in courses:
    course_students = random.sample(students, random.choice(class_size))
    registered.append([intelligence[s][0] for s in course_students])

    course_instructors = random.sample(instructors, random.choice(class_inst))
    lecturers.append([skill[i][0] for i in course_instructors])

embeddings = []
p_labels = []
labels = []
for c in courses:
    e = [
        difficulty[c][0],
        np.mean(registered[c]),
        np.sum(lecturers[c])
    ]
    embeddings.append(e)

embeddings = np.array(embeddings)
embeddings = preprocessing.scale(embeddings)

for e in embeddings:
    p = logistic(e[0] - 3 * e[1] - e[2] - 1)
    p_labels.append([p])
    labels.append([random.random() < p])


class OracleClassifier(nn.Module):
    def __init__(self):
        super(OracleClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)


class PaddedEmbeddingClassifier(torch.nn.Module):
    def __init__(self):
        super(PaddedEmbeddingClassifier, self).__init__()
        self.phi_sk = torch.nn.Linear(max(class_inst), 1)

        self.phi_i  = torch.nn.Linear(max(class_size), 1)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(3, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, input):
        offset = 1 + max(class_inst)
        sk = self.phi_sk(input[:, 1:offset])
        i = self.phi_i(input[:, offset:offset+max(class_size)])
        d = input[:, 0].reshape(-1, 1)
        processed = torch.cat((d, sk, i), dim = 1)
        return self.model(processed)

class VariableEmbeddingClassifier(nn.Module):
    def __init__(self):
        super(VariableEmbeddingClassifier, self).__init__()
        self.input_i = nn.Linear(1, 1)
        self.input_sk = nn.Linear(1, 1)

        self.output_i = nn.Linear(2, 1)
        self.output_sk = nn.Linear(2, 1)

        self.model = nn.Sequential(
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

    def forward(self, diff, skills, aptitude):
        agg_skill = torch.FloatTensor([0])
        for s in skills:
            agg_skill += self.input_sk(s.unsqueeze(dim=-1))
        count = len(skills)
        temp = torch.cat((agg_skill, torch.FloatTensor([count])), dim=-1)
        sk = self.output_sk(temp)

        agg_apt = torch.FloatTensor([0])
        for apt in aptitude:
            agg_apt += self.input_i(apt.unsqueeze(dim=-1))
        count = len(skills)
        temp = torch.cat((agg_apt, torch.FloatTensor([count])), dim=-1)
        i = self.output_i(temp)

        processed = torch.cat((diff, sk, i), dim=0)
        return self.model(processed)


labels_t = torch.FloatTensor(labels)
embeddings_t = torch.FloatTensor(embeddings)

pad_registered = pad(registered)
pad_lecturers = pad(lecturers)

pad_grounding = torch.cat((torch.FloatTensor(difficulty),
                       torch.FloatTensor(pad_registered),
                       torch.FloatTensor(pad_lecturers)), dim=1)

oracle_model = OracleClassifier()
pad_model = PaddedEmbeddingClassifier()
var_model = VariableEmbeddingClassifier()

criterion = nn.MSELoss()
oracle_optimizer = torch.optim.Adam(oracle_model.parameters(), lr=0.01)
pad_optimizer = torch.optim.Adam(pad_model.parameters(), lr=0.01)
var_optimizer = torch.optim.Adam(var_model.parameters(), lr=0.01)

'''
    Train the models
'''

print("training oracle")

for i in range(epochs):
    oracle_optimizer.zero_grad()
    oracle_output = oracle_model(embeddings_t)
    loss = criterion(oracle_output, labels_t)
    loss.backward()
    oracle_optimizer.step()
    print(f"loss: {loss:.04}, "
          f"{(oracle_output.round() == labels_t).sum()} / 5000 correct, ")


print("training padding")
for i in range(epochs):
    pad_optimizer.zero_grad()
    pad_output = pad_model(pad_grounding)
    loss = criterion(pad_output, labels_t)
    loss.backward()
    pad_optimizer.step()
    print(f"loss: {loss:.04}, "
          f"{(pad_output.round() == labels_t).sum()} / 5000 correct, ")

print("training variable")
for i in range(epochs):
    total_loss = 0
    loss = 0
    var_model.zero_grad()
    var_outputs = []
    for c in courses:
        var_outputs.append(var_model(torch.FloatTensor(difficulty[c]),
                                torch.FloatTensor(lecturers[c]),
                                torch.FloatTensor(registered[c])))
    var_output = torch.stack(var_outputs)
    loss += criterion(var_output, labels_t)
    loss.backward()
    var_optimizer.step()
    print(f"loss: {loss:.04}, "
          f"{(var_output.round() == labels_t).sum()} / 5000 correct, ")

'''
    Plot ROC curves
'''

plt.title('Receiver Operating Characteristic')

fpr, tpr, threshold = roc_curve(labels, oracle_output.detach().numpy())
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'g', label = 'Oracle, AUC = %0.2f' % roc_auc)

fpr, tpr, threshold = roc_curve(labels, pad_output.detach().numpy())
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'orange', label = 'Padded Embed Learning, AUC = %0.2f' % roc_auc)

fpr, tpr, threshold = roc_curve(labels, var_output.detach().numpy())
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'b', label = 'Variable Embed Learning, AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
