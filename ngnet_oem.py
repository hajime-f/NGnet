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


        self.mu = [2 * np.random.rand(N, 1) - 1 for i in range(M)]

        for i in range(M):
            self.Lambda.append(np.diag([random.random() for j in range(N + 1)]))

        for i in range(M):
            self.Sigma.append(self.Lambda[i][0:N, 0:N])

        for i in range(M):
            w = 2 * np.random.rand(D, N) - 1
            w_tilde = np.insert(w, np.shape(w)[1], 1.0, axis=1)
            self.W.append(w_tilde)

        self.var = [1.0 for i in range(M)]

        self.eta = 1 / ((1 + lam) / 0.9999)

        self.one = [1.0 for i in range(M)]
        self.x = [np.zeros((N, 1)) for i in range(M)]
        self.y2 = [1.0 for i in range(M)]
        self.xy = [np.zeros((N+1, D)) for i in range(M)]
        
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

        return self.W[i] @ x_tilde
    

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
        self.update_mu()
        # self.update_Lambda(x_t)
        self.update_var()


    def update_weighted_mean(self, x_t, y_t):

        self.eta = (1 + self.lam) / self.eta
        
        tmp = [x_t[j].item() for j in range(len(x_t))]
        tmp.append(1)
        x_tilde = np.array(tmp).reshape(-1, 1)
        
        for i in range(self.M):
            self.one[i] = self.one[i] + self.eta * (self.posterior_i[i] - self.one[i])
            self.x[i] = self.x[i] + self.eta * (x_t * self.posterior_i[i] - self.x[i])
            self.y2[i] = self.y2[i] + self.eta * (y_t.T @ y_t * self.posterior_i[i] - self.y2[i])
            self.xy[i] = self.xy[i] + self.eta * (x_tilde * y_t.T * self.posterior_i[i] - self.xy[i])

        for i in range(self.M):
            if np.any(np.isnan(self.x[i])):
                self.x[i] = np.zeros((self.N, 1))
            if np.any(np.isnan(self.y2[i])):
                self.y2[i] = 0.0
            if np.any(np.isnan(self.xy[i])):
                self.xy[i] = np.zeros((self.N+1, self.D))

            
    def update_mu(self):

        for i in range(self.M):
            self.mu[i] = self.x[i] / self.one[i]


    def update_Lambda(self, x_t):
        
        tmp = [x_t[j].item() for j in range(len(x_t))]
        tmp.append(1)
        x_tilde = np.array(tmp).reshape(-1, 1)

        for i in range(self.M):
            
            t1 = self.Lambda[i] @ x_tilde
            t2 = x_tilde.T @ self.Lambda[i]
            numerator = self.posterior_i[i] * t1 @ t2
            denominator = (1 / self.eta) - 1 + self.posterior_i[i] * x_tilde.T @ self.Lambda[i] @ x_tilde

            self.Lambda[i] = (1 / (1 - self.eta)) * (self.Lambda[i] - numerator / denominator)
            

    def update_var(self):

        for i in range(self.M):
            self.var[i] = (1 / self.D) * (self.y2[i] - np.trace(self.W[i] @ self.xy[i])) / self.one[i]
            
            
    
def func1(x_1, x_2):
    s = np.sqrt(np.power(x_1, 2) + np.power(x_2, 2))
    return np.sin(s) / s


if __name__ == '__main__':

    N = 2
    D = 1
    M = 20

    lam = 0.998
    
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

    pdb.set_trace()
        
    # Inference the output y
    inference_x_list = []
    for t in range(inference_T):
        inference_x_list.append(20 * np.random.rand(N, 1) - 10)
    inference_y_list = []
    for x_t in inference_x_list:    
        print(ngnet.get_output_y(x_t))


        
