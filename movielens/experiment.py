import pandas as pd
import torch
from torch import nn

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, mean_squared_error

'''
    Setup Data
'''

## load data
rating_names = ["user id", "movie", "rating", "timestamp"]
ratings = pd.read_csv("u.data",
                      sep='\t',
                      names=rating_names)

demo_names = ["user id", "age", "gender", "occupation", "zip code"]
demographics = pd.read_csv("u.user",
                           sep='|',
                           names=demo_names,
                           index_col=demo_names[0])


movie_names = ["movie id", "movie title", "release date", "video release date",
               "IMDb URL", "unknown", "Action", "Adventure", "Animation",
               "Children's", "Comedy", "Crime", "Documentary", "Drama",
               "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
               "Romance", "Sci-Fi", "Thriller", "War", "Western"]
ignored_columns = ["movie title", "video release date", "IMDb URL"]
movie_info = pd.read_csv("u.item",
                     sep='|',
                     names=movie_names,
                     usecols=set(movie_names),
                     index_col=movie_names[0],
                     encoding = "ISO-8859-1")

## pre-processing

ratings["rating"] = ratings.rating >= 4

# only ratings on or before this timestamp are used
cutoff = 884673930
ratings = ratings[ratings.timestamp <= cutoff]

# join so that movie information is part of ratings
#ratings = ratings.drop("movie", axis=1)


# ratings on or before this timestamp are used for training
train_cutoff = 880845177

train_ratings = (ratings[ratings.timestamp <= train_cutoff]
                 .drop("timestamp",axis=1))
test_ratings = (ratings[ratings.timestamp > train_cutoff]
                .drop("timestamp", axis=1))

# group ratings by user
train_ratings = train_ratings.groupby("user id")
test_ratings = test_ratings.groupby("user id")

# truth values for gender of user
train_target = demographics.loc[train_ratings.groups].gender == "F"
test_target = demographics.loc[test_ratings.groups].gender == "F"

'''
    Setup Models
'''

class LinearRegressionClassifier(nn.Module):
    def __init__(self):
        super(LinearRegressionClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)


class VariableEmbeddingClassifier(nn.Module):
    def __init__(self):
        super(VariableEmbeddingClassifier, self).__init__()
        self.input_sk = nn.Linear(1, 1)

        self.output_sk = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, v):
        mapped_v = self.input_sk(v.unsqueeze(dim=-1))\
            .sum().unsqueeze(dim=-1)
        count = len(mapped_v)
        temp = torch.cat((mapped_v, torch.Tensor([count])), dim=-1)
        return self.output_sk(temp)

n = 2
k = 5
j = 9

class VariableNNEmbeddingClassifier(nn.Module):
    def __init__(self):
        super(VariableNNEmbeddingClassifier, self).__init__()
        self.input_sk = nn.Linear(j, 1)

        self.output_sk = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, v):
        mapped_v = self.input_sk(v).sum(dim=0)
        count = len(v) * n
        temp = torch.cat((mapped_v, torch.Tensor([count])), dim=-1)
        return self.output_sk(temp)

class LookupEmbeddingClassifier(nn.Module):
    def __init__(self):
        super(LookupEmbeddingClassifier, self).__init__()

        self.e = nn.Embedding(2000, j)
        self.e.weight.data.fill_(0)
        self.s = nn.Sigmoid()

    def forward(self, movies, ratings):
        lookup = self.e((movies).long())
        return self.s(lookup.mean())

'''
    Train Models
'''

criterion = nn.BCELoss()
train_target = torch.Tensor(train_target.values.astype("int"))
test_target = torch.Tensor(test_target.values.astype("int")).unsqueeze(-1)


def ll_criterion(p, a):
    return -(a * torch.log2(p) + (1 - a) * torch.log2(1 - p)).mean()


'''
lr_model = LinearRegressionClassifier()
lr_optimizer = torch.optim.Adam(lr_model.parameters(), lr=0.005)


lr_input = torch.Tensor(train_ratings.mean().values)

for i in range(300):
    lr_optimizer.zero_grad()
    output = lr_model(lr_input).squeeze()
    loss = criterion(output, train_target)
    loss.backward()
    lr_optimizer.step()
    print(f"avg training loss: {loss:.04}")

with torch.no_grad():
    test_target = torch.Tensor(test_target.values.astype("int")).unsqueeze(-1)
    output = lr_model(torch.Tensor(test_ratings.mean().values))
    mse_loss = mean_squared_error(test_target, output)
    ll_loss = ll_criterion(output, test_target)
    print(f"avg test ll_loss: {ll_loss:.04}, mse_loss: {mse_loss:.04}")


v_model = VariableNNEmbeddingClassifier()
v_optimizer = torch.optim.Adam(v_model.parameters(), lr=0.004)
v_input = [nn.functional.pad(torch.Tensor(group["rating"].values.astype("int")),
                             (0, n - len(group) % n)) for _, group in train_ratings]
test_inp = [nn.functional.pad(torch.Tensor(group["rating"].values.astype("int")),
                                      (0, n - len(group) % n)) for _, group in test_ratings]
for i in range(150):
    v_optimizer.zero_grad()
    var_outputs = []
    for user in v_input:
        var_outputs.append(v_model(user.reshape((-1, n))))
    var_output = torch.stack(var_outputs).squeeze()
    loss = criterion(var_output, train_target)
    loss.backward()
    v_optimizer.step()
    print(f"var training loss: {loss:.04}")

with torch.no_grad():
    
    var_outputs = []
    for user in test_inp:
        var_outputs.append(v_model(user.reshape((-1, n))))
    var_output = torch.stack(var_outputs)
    mse_loss = mean_squared_error(test_target, var_output)
    ll_loss = ll_criterion(var_output, test_target)
    print(f"var test ll_loss: {ll_loss:.04}, mse_loss: {mse_loss:.04}")
'''

e_model = LookupEmbeddingClassifier()
e_optimizer = torch.optim.Adam(e_model.parameters(), lr=0.01)
e_r_input = [nn.functional.pad(torch.Tensor(group["rating"].values.astype("int")),
                             (0, n - len(group) % n)) for _, group in train_ratings]
e_m_input = [nn.functional.pad(torch.Tensor(group["movie"].values.astype("int")),
                             (0, n - len(group) % n)) for _, group in train_ratings]
e_input = [torch.stack((e_m_input[i], e_r_input[i]), dim=1).view(e_m_input[i].shape[0], -1) for i in range(len(e_r_input))]

e_tr_input = [ nn.functional.pad(torch.Tensor(group["rating"].values.astype("int")),
                          (0, n - len(group) % n)) for _, group in test_ratings]
e_tm_input = [ nn.functional.pad(torch.Tensor(group["movie"].values.astype("int")),
                          (0, n - len(group) % n)) for _, group in test_ratings]
et_input = [torch.stack((e_tm_input[i], e_tr_input[i]), dim=1).view(e_tm_input[i].shape[0], -1) for i in range(len(e_tr_input))]

for i in range(200):
    e_optimizer.zero_grad()
    e_outputs = []
    for movies, ratings in zip(e_m_input, e_r_input):
        e_outputs.append(e_model(movies, ratings))
    e_output = torch.stack(e_outputs).squeeze()
    loss = criterion(e_output, train_target)
    loss.backward()
    e_optimizer.step()
    print(f"embed training loss: {loss:.04}")

    with torch.no_grad():
        var_outputs = []
        for movies, ratings in zip(e_tm_input, e_tr_input):
            var_outputs.append(e_model(movies, ratings))
        var_output = torch.stack(var_outputs)
        mse_loss = mean_squared_error(test_target, var_output)
        ll_loss = ll_criterion(var_output, test_target)
        print(f"embed test ll_loss: {ll_loss:.04}, mse_loss: {mse_loss:.04}")

'''
    Plot ROC curves
'''

plt.title('Receiver Operating Characteristic')

fpr, tpr, threshold = roc_curve(test_target, var_output.detach().numpy())
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'blue', label = 'Embd, AUC = %0.2f' % roc_auc)


plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
