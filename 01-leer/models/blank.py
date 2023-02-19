import numpy as np


class Blank:
    def __init__(self, n=3, lr=0.01):
        # init the layers
        pass

    def fit(self, x_train, y_train):
        # fit the weights in the model to the data
        pass

    def forward(self, x):
        # pass the data x trough the model and output y
        pass

    def get_loss(self, x, y):
        # this function normally is not defined by us.
        # here we are calculating the loss we can expect
        pass


def get_model(n=3, lr=0.01):
    return Linear(n, lr)
