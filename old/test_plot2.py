from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def norm_pdf(diff, var):

    Dlog2pi = 1 * np.log(2 * np.pi)
    log_pdf1 = 1 * np.log(var)
    log_pdf2 = (1 / var) * (diff.T @ diff)
    
    return np.exp(-0.5 * (Dlog2pi + log_pdf1 + log_pdf2))


if __name__ == '__main__':    

    inference_x_list = []
    for t in range(500):
        inference_x_list.append(8 * np.random.rand(2, 1) - 4)
    inference_y_list = []
    for x_t in inference_x_list:
        inference_y_list.append(norm_pdf(x_t, 5.0))

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
    ax.plot(X1_appx, X2_appx, Y1_appx, marker="o", linestyle='None', color="r")    
    ax.set_xlabel('X1 axis')
    ax.set_ylabel('X2 axis')
    ax.set_zlabel('Y axis')
    plt.show()    


