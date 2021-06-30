from scipy.stats import multivariate_normal
import numpy as np
import random 

# This code is the implementation of the Normalized Gaussian Network (NGnet)
# In the details, see the article shown below.

# Masa-aki Sato & Shin Ishii
# On-line EM Algorithm for the Normalized Gaussian Network
# Neural Computation, 2000 Vol.12, pp.407-432, 2000
# https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.37.3704&rep=rep1&type=pdf

class NGnet:
    
    mu = []     # Center vectors of N-dimensional Gaussian functions
    Sigma = []  # Covariance matrices of N-dimensional Gaussian functions
    W = []      # Linear regression matrices in the units
    
    N  = 0      # The dimension of input data
    D = 0       # The dimension of output data
    M = 0       # The number of units

    var = []    # Covariance matrices of D-dimensional Gaussian functions
    posterior_i = []   # Posterior probability that the i-th unit is selected for each observation

    T = 0       # The number of learning data
    
    def __init__(self, N, D, M):
        
        # The constructor initializes mu, Sigma and W.
        for i in range(M):
            self.mu.append(np.array([0.0] * N))
        
        for i in range(M):
            x = np.array([1.0] * N)
            self.Sigma.append(np.diag(x))

        for i in range(M):
            w = 2 * np.random.rand(D, N) - 1
            # w = np.array([[0.5, 0.4, 0.7, 0.2], [0.2, 0.5, 0.3, 0.7], [0.3, 0.8, 0.2, 0.8]])
            w_tilde = np.insert(w, np.shape(w)[1], 1.0, axis=1)
            self.W.append(w_tilde)

        self.N = N
        self.D = D
        self.M = M

        for i in range(M):
            v = np.array([1.0] * D)
            self.var.append(np.diag(v))
        
        # for i in range(M):
        #     self.posterior_i.append([])
        

    ### The functions written below are to calculate the output y given the input x
        
    # This function returns the output y corresponding the input x.
    def get_output_y(self, x):

        # Initialization of the output vector y
        y = np.array([0.0] * self.D)

        # Calculation of Equation (2.1a) in the article
        for i in range(self.M):
            N_i = self.evaluate_Normalized_Gaussian_value(x, i)  # N_i(x)
            Wx = self.linear_regression(x, i)  # W_i * x
            
            y += (N_i * Wx)[0:self.D, 0]
        
        return y


    # This function calculates N_i(x).
    def evaluate_Normalized_Gaussian_value(self, x, i):

        sum_g_j = 0
        for j in range(self.M):
            sum_g_j += multivariate_normal.pdf(x, mean=self.mu[j], cov=self.Sigma[j])
        
        g_i = multivariate_normal.pdf(x, mean=self.mu[i], cov=self.Sigma[i])

        N_i = g_i / sum_g_j
        
        return N_i

    
    # This function calculates W_i * x.
    def linear_regression(self, x, i):

        x_tilde = np.insert(x, len(x), 1.0).reshape(-1, 1)
        Wx = np.dot(self.W[i], x_tilde)

        return Wx

    
    ### The functions written below are to learn the parameters according to the EM algorithm.

    # This function calculates equation (2.2)
    def calc_P_xyi(self, x, y, i):

        # Equation (2.3a)
        P_i = 1 / self.M

        # Equation (2.3b)
        P_x = multivariate_normal.pdf(x, mean=self.mu[i], cov=self.Sigma[i])

        # Equation (2.3c)
        diff = y.reshape(-1, 1) - self.linear_regression(x, i)
        P_y = multivariate_normal.pdf(x, mean=diff.flatten(), cov=self.var[i])

        # Equation (2.2)
        P_xyi = P_i * P_x * P_y
        
        return P_xyi
    

    # This function executes E-step written by equation (3.1)
    def offline_E_step(self, x_list, y_list):

        for x_t, y_t in zip(x_list, y_list):
            p_t = []
            for i in range(self.M):
                sum_p = 0
                for j in range(self.M):
                    sum_p += self.calc_P_xyi(x_t, y_t, j)
                p = self.calc_P_xyi(x_t, y_t, i)
                p_t.append(p / sum_p)
            self.posterior_i.append(p_t)

            
    # This function executes M-step written by equation (3.4)
    def offline_M_step(self, x_list, y_list):
        
        
        
        pass


    def offline_learning(self, x_list, y_list):
        
        if len(x_list) != len(y_list):
            print('Error: The number of input vectors x does not correspend to the number of output vectors y.')
            exit()
        else:
            self.T = len(x_list)

        self.offline_E_step(x_list, y_list)
        self.offline_M_step(x_list, y_list)
        
            
            
        

if __name__ == '__main__':

    N = 4
    D = 3
    M = 5
    
    ngnet = NGnet(N, D, M)

    # x = 2 * np.random.rand(N, 1) - 1
    x = np.array([0.4, 0.5, 0.3, 0.2]).reshape(-1, 1)

    y = ngnet.get_output_y(x)
    
