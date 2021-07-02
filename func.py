from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def mexican_hat_function(x_1, x_2):

    s = np.sqrt(np.power(x_1, 2) + np.power(x_2, 2))
    return np.sin(s) / s


if __name__ == '__main__':

    x1 = np.arange(-10.0, 10.0, 0.01)
    x2 = np.arange(-10.0, 10.0, 0.01)

    X1, X2 = np.meshgrid(x1, x2)
    Y = mexican_hat_function(X1, X2)
    
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x1, x2)")

    ax.plot_wireframe(X1, X2, Y)
    plt.show()

