import numpy as np


class MyLogisticRegression:
    def __init__(self, tol=1e-4, max_iter=100, alpha=0.01):
        self.coef_ = []
        self.intercept_ = 0
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = alpha

    # fit function   -   fit the model according to the given training data
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.m = len(y)
        self.theta = np.zeros(self.X.shape[1])
        self.alpha = 0.01
        self.cost_function()
        self.coef_ = self.theta_[:-1]
        self.intercept_ = self.theta_[-1]
        return self

    # cost function for logistic regression with gradient descent
    def cost_function(self):
        self.iter = 0
        self.cost_ = []
        while self.iter < self.max_iter:
            self.theta_ = self.theta - self.alpha * self.gradient_descent()
            self.theta = self.theta_
            self.cost_.append(self.cost())
            self.iter += 1
        return self.cost_

    # gradient descent for logistic regression
    def gradient_descent(self):
        self.iter += 1
        self.theta_ = self.theta - self.alpha * self.cost_derivative()
        return self.theta_

    # cost function for logistic regression
    def cost(self):
        self.h = self.sigmoid(self.X.dot(self.theta_))
        self.J = -(self.y.T.dot(np.log(self.h)) + (1 - self.y).T.dot(np.log(1 - self.h))) / self.m
        return self.J

    # cost derivative for logistic regression
    def cost_derivative(self):
        self.h = self.sigmoid(self.X.dot(self.theta_))
        self.J = (self.h - self.y) / self.m
        return self.J

    # sigmoid function for logistic regression
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # predict function for logistic regression using intercept and coefficients
    def predict(self, X):
        self.X = X
        self.h = self.sigmoid(self.X.dot(self.coef_) + self.intercept_)
        return np.where(self.h >= 0.5, 1, 0)
