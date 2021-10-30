from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA

def multinorm_pdf(x, mean, covinv):

    Nlog2pi = 2 * np.log(2 * np.pi)
    logdet = np.log(LA.det(covinv))
    diff = x - mean
    
    logpdf = -0.5 * (Nlog2pi - logdet + (diff.T @ covinv @ diff))
    
    return np.exp(logpdf)    
    

if __name__ == '__main__':    
    
    x1 = np.linspace(-4, 4, 500)
    x2 = np.linspace(-4, 4, 500)
    X1, X2 = np.meshgrid(x1, x2)
    pos = np.empty(X1.shape + (2,))
    pos[:, :, 0] = X1; pos[:, :, 1] = X2
    rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])

    inference_x_list = []
    for t in range(500):
        inference_x_list.append(8 * np.random.rand(2, 1) - 4)
    inference_y_list = []
    for x_t in inference_x_list:
        inference_y_list.append(multinorm_pdf(x_t, np.array([0.0, 0.0]).reshape(-1, 1), np.array([[1.0, 0.0], [0.0, 1.0]])))
    
    x1_list = []
    x2_list = []
    y1_list = []
    for x_t, y_t in zip(inference_x_list, inference_y_list):
        x1_list.append(x_t[0].item())
        x2_list.append(x_t[1].item())
        y1_list.append(y_t.item())
    X1_appx = np.array(x1_list)
    X2_appx = np.array(x2_list)
    Y1_appx = np.array(y1_list)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.plot_surface(X1, X2, rv.pdf(pos))
    ax.plot(X1_appx, X2_appx, Y1_appx, marker="o", linestyle='None', color="r")    
    ax.set_xlabel('X1 axis')
    ax.set_ylabel('X2 axis')
    ax.set_zlabel('Y axis')
    plt.show()    
    
