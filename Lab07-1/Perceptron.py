import numpy as np
import matplotlib.pyplot as plt
import csv

class Perceptron:
    def __init__(self, input_size: int, output_size: int):
        rnp = np.random.default_rng()
        self.weights = rnp.standard_normal((input_size, output_size))
        self.bias = rnp.standard_normal((output_size, 1))
        self.losses = []

    def fitting(self, X, y, epochs=1000, learning_rate=0.01):
        for _ in range(epochs):
            _y = self.predict(X)
            self.losses.append(0.5*(np.sum(np.power(_y - y, 2))))
            gradient = np.dot(X.T, _y - y) / X.shape[0]
            self.weights -= learning_rate * gradient
            self.bias -= learning_rate * np.sum(_y - y) / X.shape[0]

    def predict(self, X):
        return np.where(np.dot(X, self.weights) + self.bias > 0, 1, 0)


if __name__ == "__main__":
    with open("data.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        data = np.array(list(reader)).astype(np.float32)
    # z-score
    data[:, 0] = (data[:, 0] - data[:, 0].mean()) / data[:, 0].std()
    data[:, 1] = (data[:, 1] - data[:, 1].mean()) / data[:, 1].std()
    np.random.shuffle(data)

    data_test = data[:]
    age, estimated_salary, purchased = data_test[:, 0], data_test[:, 1], data_test[:, 2]
    age_test, estimated_salary_test, purchased_test = data_test[:, 0], data_test[:, 1], data_test[:, 2]

    X, y = data[:, :-1], data[:, -1]
    y = y.reshape(-1, 1)
    model = Perceptron(X.shape[1], 1)
    model.fitting(X, y)

    X_test = np.column_stack((age_test, estimated_salary_test))
    y_test = np.expand_dims(purchased_test, axis=1)
    _y = model.predict(X_test)

    count_correct = 0
    for i in range(len(_y)):
        if _y[i] == y_test[i]:
            print(f'\033[92mPrediction: {_y[i][0]:.2f}, Actual: {y_test[i][0]}\033[0m')
            count_correct += 1
        else:
            print(f'\033[91mPrediction: {_y[i][0]:.2f}, Actual: {y_test[i][0]}\033[0m')
    print(f'Accuracy: {count_correct / len(_y):.2%}')

    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')

    for i, (age, estimated_salary, purchased) in enumerate(data_test):
        if y_test[i] == 1:
            plt.scatter(age, estimated_salary, color='red')
        else:
            plt.scatter(age, estimated_salary, color='blue')
    plt.savefig('data/img/perceptron_data.png')

    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = model.predict(np.column_stack((X.ravel(), Y.ravel()))).reshape(X.shape)
    plt.contourf(X, Y, Z, levels=1, alpha=0.5)
    plt.savefig('data/img/perceptron_prediction.png')

    plt.clf()

    plt.plot(model.losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('data/img/perceptron_loss.png')
