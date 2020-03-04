import numpy as np


class Perceptron:
    def __init__(self, x, y, w, alpha):
        self.w = w
        self.x = x
        self.y = y
        self.alpha = alpha

    def find_error_sample(self):
        sign = np.sign(np.dot(self.x, self.w))
        error_index = np.where((self.y - sign) != 0)[0]
        return error_index

    def loss(self):
        sign = np.sign(np.dot(self.x, self.w))
        return int(np.dot(self.y.T, sign))

    def train(self):
        while len(self.find_error_sample()) > 0:

            random_index = np.random.choice(self.find_error_sample())
            random_x = x[random_index]
            random_y = y[random_index]
            self.w = self.w + self.alpha * (random_y * random_x).reshape(-1, 1)
            print(self.w)

        return self.w


if __name__ == '__main__':
    x1 = np.array([[3, 3],
                   [4, 3],
                   [1, 1]])
    x0 = np.ones([3, 1])
    x = np.hstack([x1, x0])

    y = np.array([1, 1, -1]).reshape(-1, 1)

    alpha = 1
    w = np.array([0, 0, 0]).reshape(-1, 1)
    perceptorn = Perceptron(x, y, w, alpha)
    weight = perceptorn.train()
