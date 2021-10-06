## imports

import pandas as pd
from sklearn import model_selection, preprocessing, metrics
import argparse
import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from pathlib2 import Path
from typing import List, Tuple

## data - custom code needed
class MovieDataset:
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, item):
        user = self.users[item]
        movie = self.movies[item]
        rating = self.ratings[item]

        x = torch.tensor([user, movie], dtype=torch.long)
        y = torch.tensor([rating], dtype=torch.float)

        return x, y


## network - custom code needed
class Net(nn.Module):
    def __init__(self, num_users, num_movies):
        super().__init__()

        self.user_embed = nn.Embedding(num_users, 32)
        self.movie_embed = nn.Embedding(num_movies, 32)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        user_embeds = self.user_embed(x[:, 0])
        movie_embeds = self.movie_embed(x[:, 1])
        output = torch.cat([user_embeds, movie_embeds], dim=1)
        output = self.output(output)

        return output


## loss function and model update
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)


## training loop
def fit(args, model, loss_func, opt, scheduler, train_dl, valid_dl):

    # loop over epochs
    for epoch in range(args.epochs):

        # train step
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        # eval steps
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        scheduler.step(val_loss)
        print(epoch, val_loss)


## data loader function for training and validation
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


## main function - custom code needed
if __name__ == "__main__":

    # Training Arguments
    parser = argparse.ArgumentParser(description="PyTorch Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="number of epochs to train (default: 20)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        metavar="LR",
        help="learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model", type=bool, default=False, help="For Saving the current Model"
    )
    args = parser.parse_args()

    # GPU / CPU set up
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # Read in data
    df = pd.read_csv("./data/ml-latest-small/ratings.csv")
    lbl_user = preprocessing.LabelEncoder()
    lbl_movie = preprocessing.LabelEncoder()

    df.userId = lbl_user.fit_transform(df.userId.values)
    df.movieId = lbl_movie.fit_transform(df.movieId.values)

    df_train, df_valid = model_selection.train_test_split(
        df, test_size=0.1, random_state=42, stratify=df.rating.values
    )

    train_dataset = MovieDataset(
        users=df_train.userId.values,
        movies=df_train.movieId.values,
        ratings=df_train.rating.values,
    )
    valid_dataset = MovieDataset(
        users=df_valid.userId.values,
        movies=df_valid.movieId.values,
        ratings=df_valid.rating.values,
    )
    train_dl, valid_dl = get_data(train_dataset, valid_dataset, args.batch_size)

    # Model
    model = Net(
        num_users=len(lbl_user.classes_), num_movies=len(lbl_movie.classes_)
    ).to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(model)
    n = count_parameters(model)
    print("Number of parameters: %s" % n)
    summary(model, input_size=(args.batch_size, 1, 28, 28))

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Loss Function
    loss_func = nn.MSELoss()
    scheduler = ReduceLROnPlateau(
        optimizer, "min", patience=5, min_lr=args.lr / 100.0, verbose=True
    )
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Train the model!!!
    fit(args, model, loss_func, optimizer, scheduler, train_dl, valid_dl)

    # Save
    if args.save_model:
        torch.save(model.state_dict(), "saved_model.pt")
