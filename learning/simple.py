import argparse
import json

import torch
from sklearn.metrics import accuracy_score

from learning.models import PhiNet

# partition factor
PRT = 0.9

parser = argparse.ArgumentParser(description='Try to learn embeddings',
                                 formatter_class=argparse.
                                    ArgumentDefaultsHelpFormatter)

parser.add_argument('-e', '--epochs', default=100, type=int,
                     help='Number of epochs to train for')
parser.add_argument('instance', metavar = "instance.json",
                    help="Location of DB instance JSON file",
                    type=argparse.FileType('r'))


def build_dataset(instance):
    data = {'difficulty': [], 'teaching_skills': [], 'student_aptitude': []}
    answers = []

    for c in instance["entities"]["courses"]:
        answers.append([c, "yes"] in instance["attributes"]["tutoring"])
        data['difficulty'].append([c, "high"] in instance["attributes"]["difficulty"])
        profs = [t[0] for t in filter(lambda t: t[1] == c,
                                             instance["relations"]["teaches"])]
        profs_skills = [[p, "high"] in instance['attributes']['teaching_skills']
                        for p in profs]
        data['teaching_skills'].append(profs_skills)
        students = [t[0] for t in filter(lambda t: t[1] == c,
                                             instance["relations"]["registered"])]
        student_apt = [[s, "high"] in instance['attributes']['intelligence']
                        for s in students]
        data['student_aptitude'].append(student_apt)

    return data, answers


def pad(data):
    N = max([len(i) for i in data['teaching_skills']])
    data['teaching_skills'] = [a + [0] * (N - len(a)) for a
                               in data['teaching_skills']]

    M = max([len(i) for i in data['student_aptitude']])
    data['student_aptitude'] = [a + [0] * (M - len(a)) for a
                               in data['student_aptitude']]

    return N, M


def partition(data):
    cutoff = round(len(data['difficulty']) * PRT)
    train_data = {
        "difficulty": data['difficulty'][:cutoff],
        "teaching_skills": data['teaching_skills'][:cutoff],
        "student_aptitude": data['student_aptitude'][:cutoff]
    }

    test_data = {
        "difficulty": data['difficulty'][cutoff:],
        "teaching_skills": data['teaching_skills'][cutoff:],
        "student_aptitude": data['student_aptitude'][cutoff:]
    }

    return train_data, test_data


def train(model, data, labels, epochs):
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for e in range(epochs):
        all_losses = []
        for i in range(len(data)):
            optimizer.zero_grad()
            output = model(torch.Tensor([data['difficulty'][i]]),
                           torch.Tensor(data['teaching_skills'][i]),
                           torch.Tensor(data['student_aptitude'][i]))
            loss = criterion(output, torch.Tensor([labels[i]]))
            all_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f"Epoch {e}/{epochs}, loss: {sum(all_losses) / len(all_losses):.3f}")


def main():
    args = parser.parse_args()
    instance = json.load(args.instance)
    data, answers = build_dataset(instance)
    dim = pad(data)
    model = PhiNet(1, dim)

    # partition the dataset
    train_data, test_data = partition(data)
    train_answers =  answers[:round(len(answers)*PRT)]
    test_answers = answers[round(len(answers)*PRT):]

    train(model, train_data, train_answers, args.epochs)

    results = model(torch.Tensor([train_data['difficulty']]),
                    torch.Tensor(train_data['teaching_skills']),
                    torch.Tensor(train_data['student_aptitude']))
    train_accuracy = accuracy_score(train_answers, results.detach().numpy().round())
    print(f"Train Accuracy: {train_accuracy}")

    results = model(torch.Tensor([test_data['difficulty']]),
                           torch.Tensor(test_data['teaching_skills']),
                           torch.Tensor(test_data['student_aptitude']))
    accuracy = accuracy_score(test_answers, results.detach().numpy().round())

    print(f"Accuracy: {accuracy}")
    print(f"Courses that actually have tutoring: {sum(test_answers)}")
    print(f"Teaching skills weights: {list(model.summarize[0].weight.detach())}")
    print(f"Student intelligence weights: {list(model.summarize[1].weight.detach())}")


# execute only if run as a script
if __name__ == "__main__":
    main()
