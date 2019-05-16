import pandas as pd
import torch
from torch import nn

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
                           names=demo_names)

movie_names = ["movie id", "movie title", "release date", "video release date",
               "IMDb URL", "unknown", "Action", "Adventure", "Animation",
               "Children's", "Comedy", "Crime", "Documentary", "Drama",
               "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
               "Romance", "Sci-Fi", "Thriller", "War", "Western"]
movies = pd.read_csv("u.item",
                     sep='|',
                     names=movie_names,
                     encoding = "ISO-8859-1")

## pre-processing

ratings["rating"] = ratings.rating >= 4

# only ratings on or before this timestamp are used
cutoff = 884673930

ratings = ratings[ratings.timestamp <= cutoff]

# ratings on or before this timestamp are used for training
train_cutoff = 880845177

train_ratings = (ratings[ratings.timestamp <= train_cutoff]
                 .drop("timestamp",axis=1))
test_ratings = (ratings[ratings.timestamp > train_cutoff]
                .drop("timestamp", axis=1))

'''
    Setup Models
'''

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
        agg_skill = self.input_sk(skills.unsqueeze(dim=-1))\
            .sum().unsqueeze(dim=-1)
        count = len(skills)
        temp = torch.cat((agg_skill, torch.Tensor([count])), dim=-1)
        sk = self.output_sk(temp)

        agg_apt = self.input_i(aptitude.unsqueeze(dim=-1))\
            .sum().unsqueeze(dim=-1)
        count = len(skills)
        temp = torch.cat((agg_apt, torch.Tensor([count])), dim=-1)
        i = self.output_i(temp)

        processed = torch.cat((diff, sk, i), dim=0)
        return self.model(processed)

'''
    Train Models
'''

print("hello")
