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

    eta = []
    lam = 0
    
    posterior_i = []   # Posterior probability that the i-th unit is selected for each observation
    
    def __init__(self, N, D, M, lam):


        self.mu.append([2 * np.random.rand(N, 1) - 1 for i in range(M)])

        p = []
        for i in range(M):
            p.append(np.diag([random.random() for j in range(N + 1)]))
        self.Lambda.append(p)
        
        p = []
        for i in range(M):
            p.append(self.Lambda[0][i][0:N, 0:N])
        self.Sigma.append(p)

        p = []
        for i in range(M):
            w = 2 * np.random.rand(D, N) - 1
            p.append(np.insert(w, np.shape(w)[1], 1.0, axis=1))
        self.W.append(p)

        self.var.append([1.0 for i in range(M)])
        self.eta.append(0.01)
        
        self.one.append([random.random() for i in range(M)])
        self.x.append([2 * np.random.rand(N, 1) - 1 for i in range(M)])
        self.y2.append([random.random() for i in range(M)])
        self.xy.append([random.random() for i in range(M)])

        self.N = N
        self.D = D
        self.M = M
        self.lam = lam

    
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
    def linear_regression(self, x_t, i):

        tmp = [x_t[j].item() for j in range(len(x_t))]
        tmp.append(1)
        x_tilde = np.array(tmp).reshape(-1, 1)
        W = self.W[-1][i]

        return W @ x_tilde
    

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

        mu = self.mu[-1][i]
        Sigma = self.Sigma[-1][i]
        
        # Equation (2.3b)
        P_x = multivariate_normal.pdf(x_t.flatten(), mu.flatten(), Sigma)

        # Equation (2.3c)
        diff = y_t.reshape(-1, 1) - self.linear_regression(x_t, i)
        P_y = self.norm_pdf(diff, self.var[-1][i])

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

        one, x, y2, xy = self.one[-1], self.x[-1], self.y2[-1], self.xy[-1]
        posterior_i = self.posterior_i

        eta = (1 + self.lam) / self.eta[-1]
        self.eta.append(eta)
        
        tmp = [x_t[j].item() for j in range(len(x_t))]
        tmp.append(1)
        x_tilde = np.array(tmp).reshape(-1, 1)

        p1, p2, p3, p4 = [], [], [], []
        for i in range(self.M):
            p1.append(one[i] + eta * (posterior_i[i] - one[i]))
            p2.append(x[i] + eta * (x_t * posterior_i[i] - x[i]))
            p3.append(y2[i] + eta * (y_t.T @ y_t * posterior_i[i] - y2[i]))
            p4.append(xy[i] + eta * (x_tilde @ y_t.T * posterior_i[i] - xy[i]))
        self.one.append(p1)
        self.x.append(p2)
        self.y2.append(p3)
        self.xy.append(p4)

        if len(one) > 100:
            del self.one[0]
            del self.x[0]
            del self.y2[0]
            del self.xy[0]
            del self.eta[0]

    def update_mu(self):
        pass
            
    
def func1(x_1, x_2):
    s = np.sqrt(np.power(x_1, 2) + np.power(x_2, 2))
    return np.sin(s) / s


if __name__ == '__main__':

    N = 2
    D = 1
    M = 20

    lam = 0.01
    
    learning_T = 1000
    inference_T = 1000
    
    ngnet = NGnet_OEM(N, D, M, lam)

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


        
