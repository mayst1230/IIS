import numpy as np
from sklearn.datasets import make_circles
from sklearn.linear_model import Perceptron
from matplotlib import pyplot as plt, pylab
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = make_circles(noise=0.2, factor=0.5, random_state=1)
x = np.array(data[0])
y = np.array(data[1])

x = StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
cm_bright = ListedColormap(['#00ff00', '#0000ff'])


def __perceptron__(x_train, y_train, x_test):
    model = Perceptron()
    model.fit(x_train, y_train.ravel())
    print('Средняя точность персептрона:', model.score(x_train, y_train))
    return model.predict(x_test)


def __10_perceptron__(x_train, y_train, x_test):
    model = MLPClassifier(hidden_layer_sizes=10, alpha=0.01)
    model.fit(x_train, y_train)
    print('Средняя точность многослойного персептрона (10 слоёв) :', model.score(x_train, y_train))
    return model.predict(x_test)


def __100_perceptron__(x_train, y_train, x_test):
    model = MLPClassifier(hidden_layer_sizes=100, alpha=0.01)
    model.fit(x_train, y_train)
    print('Средняя точность многослойного персептрона (100 слоёв):', model.score(x_train, y_train))
    return model.predict(x_test)


def paint_graph():
    y_perceptron = __perceptron__(x_train, y_train, x_test)
    y_perceptron_10 = __10_perceptron__(x_train, y_train, x_test)
    y_perceptron_100 = __100_perceptron__(x_train, y_train, x_test)
    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 2)
    ax3 = plt.subplot(2, 3, 4)
    ax4 = plt.subplot(2, 3, 5)
    ax5 = plt.subplot(2, 3, 6)
    ax1.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_bright)
    ax2.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright)
    ax3.scatter(x_test[:, 0], x_test[:, 1], c=y_perceptron, cmap=cm_bright)
    ax4.scatter(x_test[:, 0], x_test[:, 1], c=y_perceptron_10, cmap=cm_bright)
    ax5.scatter(x_test[:, 0], x_test[:, 1], c=y_perceptron_100, cmap=cm_bright)
    fig = pylab.gcf()
    fig.canvas.set_window_title('ПИбд-42 Майоров - 8 вариант')
    plt.show()


def main():
    paint_graph()


if __name__ == '__main__':
    main()
