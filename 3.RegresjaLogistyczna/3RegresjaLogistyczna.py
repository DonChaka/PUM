from typing import Callable

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from funkcje import PUMData
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def mse_cost(predicted, actual):
    return np.mean((predicted - actual) ** 2)


def logistic_cost(predicted, actual):
    return -np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))


class MyLogisticRegression:
    def __init__(self, learning_rate: float = 0.05, max_iterations: int = 100, min_cost_diff: float = 1e-4,
                 cost_function: Callable = mse_cost):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.min_cost_diff = min_cost_diff
        self.coef_ = None
        self.intercept_ = None
        self.cost_function = cost_function
        self.costs = []

    def fit(self, X, Y):
        if self.coef_ is None:
            self.coef_ = np.random.normal(loc=1, scale=.15, size=(1, X.shape[1]))
            self.intercept_ = np.random.normal(loc=1, scale=0.15, size=1)

        Y = Y.reshape(-1, 1)
        n = X.shape[0]
        for i in range(self.max_iterations):
            preds = sigmoid(X.dot(self.coef_.T) + self.intercept_)
            cost = self.cost_function(preds, Y)

            dW = 1/n * np.dot((preds - Y).T, X)
            dB = 1/n * np.sum(preds - Y)

            self.coef_ = self.coef_ - self.learning_rate * dW
            self.intercept_ = self.intercept_ - self.learning_rate * dB

            self.costs.append(cost)

            if len(self.costs) >= 2 and self.costs[-2] - self.costs[-1] < self.min_cost_diff:
                break

            if len(self.costs) >= 2 and self.costs[-1] > self.costs[-2]:
                self.learning_rate /= 2

        return self

    def decision_function(self, X):
        return sigmoid(X.dot(self.coef_.T) + self.intercept_)

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0.5, 1, 0)


random_state = 244827
n_samples = 2427

data = PUMData(make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, random_state=random_state))

my_model = MyLogisticRegression(learning_rate=0.5, max_iterations=100, min_cost_diff=0.0001, cost_function=logistic_cost)

my_model.fit(data.x_train, data.y_train)

sk_model = LogisticRegression(max_iter=100, random_state=random_state)
sk_model.fit(data.x_train, data.y_train)

print(f'Moj model: {my_model.coef_} + [{my_model.intercept_}]')
print(f'Sklearn model: {sk_model.coef_} + {sk_model.intercept_}')

plt.plot(my_model.costs)
plt.show()

x_min, x_max = data.x_test[:, 0].min() - 0.25, data.x_test[:, 0].max() + 0.25
y_min, y_max = data.x_test[:, 1].min() - 0.25, data.x_test[:, 1].max() + 0.25
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z_an = my_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z_sk = sk_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

f, axarr = plt.subplots(1, 2, sharex="col", sharey="row", figsize=(20, 15))

for idx, Z, title in zip(
    range(2),
    [Z_an, Z_sk],
    ['Implementacja własna', 'Model LogisticRegression']):
    axarr[idx].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx].scatter(data.x_test[:, 0], data.x_test[:, 1], c=data.y_test, s=20, edgecolor="k")
    axarr[idx].set_title(title)

plt.show()

y_an_predicted = my_model.predict(data.x_test)
y_sk_predicted = sk_model.predict(data.x_test)

cm_sk = confusion_matrix(data.y_test, y_sk_predicted)
cm_an = confusion_matrix(data.y_test, y_an_predicted)


f, axarr = plt.subplots(1, 2, sharex="col", sharey="row", figsize=(15, 8))

for idx, cm, title in zip(
    range(2),
    [cm_an, cm_sk],
    ['Implementacja własna', 'Model RidgeClassifier'],
):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sk_model.classes_)
    axarr[idx].set_title(title)
    disp.plot(ax=axarr[idx])

plt.show()