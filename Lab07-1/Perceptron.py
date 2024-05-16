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

class MultiPerceptron:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        rnp = np.random.default_rng()
        self.weights_1 = rnp.standard_normal((input_size, hidden_size))
        self.bias_1 = rnp.standard_normal((1, hidden_size))
        self.weights_2 = rnp.standard_normal((hidden_size, output_size))
        self.bias_2 = rnp.standard_normal((1, output_size))
        self.losses = []

    def fitting(self, X, y, epochs=1000, learning_rate=0.01):
        w1 = np.vstack((self.weights_1, self.bias_1))
        w2 = np.vstack((self.weights_2, self.bias_2))
        bias = np.ones((X.shape[0], 1))

        for _ in range(epochs):
            hidden_input = np.dot(np.column_stack((X, bias)), w1)
            hidden_output = self.sigmoid(hidden_input)
            outer_input = np.dot(np.column_stack((hidden_output, bias)), w2)
            output = self.sigmoid(outer_input)

            self.losses.append(0.5*(np.sum(np.power((output - y), 2))))

            # backward pass (output layer)
            # delta(E_total) /  delta(output_o)
            gradient_cal1 = (output - y)  # data_size x output_layer_size
            # delta(output_o) / delta(net_o)
            gradient_cal2 = self.sigmoid_derivative(output)  # data_size x output_layer_size
            # delta(net_o) / delta(w2)
            gradient_cal3 = np.column_stack((hidden_output, bias))  # data_size x (hidden_layer_size + 1)
            # delta(E_total) / delta(w2)
            gradient_cal4 = np.dot(gradient_cal3.T, gradient_cal1 * gradient_cal2)
            # (hidden_layer_size + 1) x output_layer_size
            # update w2
            weights2_update = gradient_cal4 * learning_rate
            w2 -= weights2_update

            # backward pass (hidden layer)
            # delta(E_total) / delta(output_h)
            gradient_cal5 = np.dot(gradient_cal1 * gradient_cal2, w2[:-1].T)  # data_size x hidden_layer_size
            # delta(output_h) / delta(net_h)
            gradient_cal6 = self.sigmoid_derivative(hidden_output)  # data_size x hidden_layer_size
            # delta(net_h) / delta(w1)
            gradient_cal7 = np.column_stack((X, bias))  # data_size x (input_size + 1)
            # delta(E_total) / delta(w1)
            gradient_cal8 = np.dot(gradient_cal7.T, gradient_cal5 * gradient_cal6)
            # (input_size + 1) x hidden_layer_size
            # update w1
            weights1_update = gradient_cal8 * learning_rate
            w1 -= weights1_update

            self.weights_1 = w1[:-1]
            self.weights_2 = w2[:-1]
            self.bias_1 = w1[-1]
            self.bias_2 = w2[-1]

    def predict(self, X):
        hidden_input = np.dot(X, self.weights_1) + self.bias_1
        hidden_output = self.sigmoid(hidden_input)
        outer_input = np.dot(hidden_output, self.weights_2) + self.bias_2
        output = self.sigmoid(outer_input)
        return output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)




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

    # model = Perceptron(X.shape[1], 1)
    # model.fitting(X, y)

    multi_model = MultiPerceptron(X.shape[1], 4, 1)
    multi_model.fitting(X, y, epochs=1000, learning_rate=0.01)

    X_test = np.column_stack((age_test, estimated_salary_test))
    y_test = np.expand_dims(purchased_test, axis=1)
    # _y = model.predict(X_test)
    _y = multi_model.predict(X_test)

    count_correct = 0
    for i in range(len(_y)):
        # if _y[i] == y_test[i]:
        if _y[i] > 0.5 and y_test[i] == 1 or _y[i] <= 0.5 and y_test[i] == 0:
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
    # Z = model.predict(np.column_stack((X.ravel(), Y.ravel()))).reshape(X.shape)
    Z = multi_model.predict(np.column_stack((X.ravel(), Y.ravel()))).reshape(X.shape)
    plt.contourf(X, Y, Z, levels=1, alpha=0.5)
    plt.savefig('data/img/perceptron_prediction.png')

    plt.clf()

    # plt.plot(model.losses)
    plt.plot(multi_model.losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('data/img/perceptron_loss.png')
