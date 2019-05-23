import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_curve, auc, mean_squared_error, log_loss

from movielens.models import PartitioningNN, EmbeddingSum

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

'''
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
'''

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

n = 10
epochs = 300

train_target = torch.Tensor(train_target.values.astype("int"))
test_target = torch.Tensor(test_target.values.astype("int")).unsqueeze(-1)

user_ratings = [nn.functional.pad(torch.Tensor(group["rating"].values.astype("int")),
                             (0, n - len(group) % n)) for _, group in train_ratings]
user_ratings_test = [nn.functional.pad(torch.Tensor(group["rating"].values.astype("int")),
                             (0, n - len(group) % n)) for _, group in test_ratings]


user_movies = [nn.functional.pad(torch.Tensor(group["movie"].values.astype("int")),
                             (0, n - len(group) % n)) for _, group in train_ratings]
user_movies_test = [nn.functional.pad(torch.Tensor(group["movie"].values.astype("int")),
                             (0, n - len(group) % n)) for _, group in test_ratings]


'''
    Train Models
'''

def train(model, inputs, target):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for i in range(epochs):
        optimizer.zero_grad()
        output = model.forward_all(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(f"training loss: {loss:.03}")

def ll_criterion(a, p):
    return -(a * torch.log2(p) + (1 - a) * torch.log2(1 - p)).mean()

def test(model, input, target):
    with torch.no_grad():
        output = model.forward_all(input)
        mse_loss = mean_squared_error(target, output)
        ll_loss = ll_criterion(target, output)
        print(f"test ll_loss: {ll_loss:.03}, mse_loss: {mse_loss:.03}")


print("Partitioning Neural Network")
partitioningNN = PartitioningNN.PartitioningNN(n)
train(partitioningNN, user_ratings, train_target)
test(partitioningNN, user_ratings_test, test_target)

print("Embedding Sum")
embeddingSum = EmbeddingSum.EmbeddingSum()
train(embeddingSum, user_movies, train_target)
test(embeddingSum, user_movies_test, test_target)

