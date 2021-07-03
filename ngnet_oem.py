from scipy.stats import multivariate_normal
import numpy as np
import numpy.linalg as LA
import random
import pdb

# This code is the implementation of the Normalized Gaussian Network (NGnet)
# In the details, see the article shown below.

# Masa-aki Sato & Shin Ishii
# On-line EM Algorithm for the Normalized Gaussian Network
# Neural Computation, 2000 Vol.12, pp.407-432, 2000
# https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.37.3704&rep=rep1&type=pdf

class NGnet_OEM:

    mu = []      # Center vectors of N-dimensional Gaussian functions
    Sigma = []   # Covariance matrices of N-dimensional Gaussian functions
    Lambda = []  # Auxiliary variable to calculate covariance matrix Sigma
    W = []       # Linear regression matrices in units
    var = []     # Variance of D-dimensional Gaussian functions
    
    N = 0        # Dimension of input data
    D = 0        # Dimension of output data
    M = 0        # Number of units

    one = []
    x = []
    y2 = []
    xy = []

    n_one = []
    n_x = []
    n_y2 = []
    n_xy = []

    eta = 0
    
    posterior_i = []   # Posterior probability that the i-th unit is selected for each observation
    
    def __init__(self, N, D, M, eta):
        
        for i in range(M):
            self.mu.append(2 * np.random.rand(N, 1) - 1)

        for i in range(M):
            self.Lambda.append(np.diag([random.random() for j in range(N + 1)]))
            
        for i in range(M):
            self.Sigma.append(self.Lambda[i][0:N, 0:N])

        for i in range(M):
            w = 2 * np.random.rand(D, N) - 1
            w_tilde = np.insert(w, np.shape(w)[1], 1.0, axis=1)
            self.W.append(w_tilde)

        for i in range(M):
            self.var.append(1.0)

        for i in range(M):
            self.one.append(0.0)
            self.x.append(0.0)
            self.y2.append(0.0)
            self.xy.append(0.0)

        for i in range(M):
            self.n_one.append(0.0)
            self.n_x.append(0.0)
            self.n_y2.append(0.0)
            self.n_xy.append(0.0)
            
        self.N = N
        self.D = D
        self.M = M
        self.eta = eta

    
    ### The functions written below are to calculate the output y given the input x
        
    # This function returns the output y corresponding the input x according to equation (2.1a)
    def get_output_y(self, x):

        # Initialization of the output vector y
        y = np.array([0.0] * self.D).reshape(-1, 1)

        # Equation (2.1a)
        for i in range(self.M):
            N_i = self.evaluate_normalized_Gaussian_value(x, i)  # N_i(x)
            Wx = self.linear_regression(x, i)  # W_i * x
            y += N_i * Wx
        
        return y

    
    # This function calculates N_i(x) according to equation (2.1b)
    def evaluate_normalized_Gaussian_value(self, x, i):

        # Denominator of equation (2.1b)
        sum_g_j = 0
        for j in range(self.M):
            sum_g_j += multivariate_normal.pdf(x.flatten(), self.mu[j].flatten(), self.Sigma[j])

        # Numerator of equation (2.1b)
        g_i = multivariate_normal.pdf(x.flatten(), self.mu[i].flatten(), self.Sigma[i])

        # Equation (2.1b)
        N_i = g_i / sum_g_j
        
        return N_i        


    # This function calculates W_i * x.
    def linear_regression(self, x, i):

        tmp = [x[j].item() for j in range(len(x))]
        tmp.append(1)
        
        x_tilde = np.array(tmp).reshape(-1, 1)
        Wx = self.W[i] @ x_tilde

        return Wx
    

    def online_learning(self, x_t, y_t):

        self.posterior_i = []
        
        self.E_step(x_t, y_t)
        self.M_step(x_t, y_t)


    def E_step(self, x_t, y_t):

        p = []
        for i in range(self.M):
            p.append(self.calc_P_xyi(x_t, y_t, i).item())
        p_sum = sum(p)
        
        for i in range(self.M):
            self.posterior_i.append(p[i] / p_sum)
        

    # This function calculates equation (2.2)
    def calc_P_xyi(self, x_t, y_t, i):

        # Equation (2.3b)
        P_x = multivariate_normal.pdf(x_t.flatten(), self.mu[i].flatten(), self.Sigma[i])

        # Equation (2.3c)
        diff = y_t.reshape(-1, 1) - self.linear_regression(x_t, i)
        P_y = self.norm_pdf(diff, self.var[i])

        return (P_x * P_y) / self.M


    # This function calculates normal function according to equation (2.3c)
    def norm_pdf(self, diff, var):
        
        log_pdf1 = - self.D/2 * np.log(2 * np.pi)
        log_pdf2 = - self.D/2 * np.log(var)
        log_pdf3 = - (1/(2 * var)) * (diff.T @ diff)
        return np.exp(log_pdf1 + log_pdf2 + log_pdf3)

    
    def M_step(self, x_t, y_t):
        
        self.update_weighted_mean(x_t, y_t)


    def update_weighted_mean(self, x_t, y_t):

        for i in range(self.M):
            self.n_one[i] = self.one[i] + self.eta * (self.posterior_i[i] - self.one[i])
            


    
def func1(x_1, x_2):
    s = np.sqrt(np.power(x_1, 2) + np.power(x_2, 2))
    return np.sin(s) / s


if __name__ == '__main__':

    N = 2
    D = 1
    M = 20

    eta = 0.01
    
    learning_T = 1000
    inference_T = 1000
    
    ngnet = NGnet_OEM(N, D, M, eta)

    # Preparing for learning data
    learning_x_list = []
    for t in range(learning_T):
        learning_x_list.append(20 * np.random.rand(N, 1) - 10)
    learning_y_list = []
    for x_t in learning_x_list:
        learning_y_list.append(np.array(func1(x_t[0], x_t[1])))
    
    # Training NGnet
    for x_t, y_t in zip(learning_x_list, learning_y_list):
        ngnet.online_learning(x_t, y_t)
    
    # Inference the output y
    inference_x_list = []
    for t in range(inference_T):
        inference_x_list.append(20 * np.random.rand(N, 1) - 10)
    inference_y_list = []
    for x_t in inference_x_list:    
        print(ngnet.get_output_y(x_t))


        
