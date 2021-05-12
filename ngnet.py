import numpy as np
import cupy as cp
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
    var = []

    P_i = []    # Probability of i: P(i | theta)
    P_x = []    # Probability of x: P(x | i, theta)
    P_y = []    # Probability of y: P(y | x, i, theta)
    P_xyi = []  # Probability of xyi: P(x, y, i | theta)
    
    
    posterior_y = 0  # Probability of y: P(y | x, theta) that the output value becomes y
    posterior_i = []  # Probability of y: P(i | x, y, theta) that the output value becomes y
    
    N  = 0      # The dimension of input data,
    D = 0       # The dimension of output data and
    M = 0       # The number of units,
    
    def __init__(self, N, D, M):
        
        # The constructor initializes mu, Sigma and W.
        for i in range(M):
            self.mu.append(2 * np.random.rand(N, 1) - 1)

        for i in range(M):
            x = [random.random() for i in range(N)]
            self.Sigma.append(np.diag(x))

        for i in range(M):
            w = 2 * np.random.rand(D, N) - 1
            self.W.append(np.insert(w, np.shape(w)[1], 1, axis=1))

        for i in range(M):            
            self.var.append(random.random())

        P_i = [1/M] * M
        P_x = [0.1] * M
        P_y = [0.1] * M
        P_xyi = [0.1] * M
        posterior_i = [0.1] * M
        
        self.N = N
        self.D = D
        self.M = M
    

    def logpdf(self, x, mean, cov):

        # "eigh" assumes the matrix is Hermitian.
        vals, vecs = np.linalg.eigh(cov)
        logdet = np.sum(np.log(vals))
        valsinv = np.array([1./v for v in vals])
        
        # "vecs" is R times D while "vals" is a R-vector where R is the matrix rank.
        # The asterisk performs element-wise multiplication.
        U = vecs * np.sqrt(valsinv)
        rank = len(vals)
        dev = x - mean

        # "maha" for "Mahalanobis distance".
        maha = np.square(np.dot(dev.T, U)).sum()
        log2pi = np.log(2 * np.pi)
        
        return -0.5 * (rank * log2pi + maha + logdet)
    
    
    # This function calculate G_i(x).
    def evaluate_Gaussian_value(self, x, i):

        # http://gregorygundersen.com/blog/2019/10/30/scipy-multivariate/
        return np.exp(self.logpdf(x, self.mu[i], self.Sigma[i]))

    # This function calculate N_i(x).
    def evaluate_Normalized_Gaussian_value(self, x, i):

        t = self.evaluate_Gaussian_value(x, i)

        s = 0
        for j in range(self.M):
            if i == j:
                s += t
            else:
                s += self.evaluate_Gaussian_value(x, j)

        return s / t

    # This function calculate W_i * x.
    def linear_regression(self, x, i):
        
        return np.dot(self.W[i], np.insert(x, len(x), 1).reshape(-1, 1))
        
    
    # This function returns the output y corresponding the input x.
    def obtain_output(self, x):

        y = np.array([0.0] * self.M).reshape(-1, 1)
        for i in range(self.M):

            ng_value = self.evaluate_Normalized_Gaussian_value(x, i)
            lr_matrix = self.linear_regression(x, i)
            y += ng_value * lr_matrix
        
        return y


    # This function calculate a square of the Mahalanobis distance.
    def calc_maha(self, x, mean, cov):
        
        vals, vecs = np.linalg.eigh(cov)
        valsinv = np.array([1./v for v in vals])
        
        U = vecs * np.sqrt(valsinv)
        dev = x - mean

        maha = np.square(np.dot(dev.T, U)).sum()

    def calc_recip_var(self, var, val):
        return np.reciprocal(np.power(var, val))

    def calc_recip_pi(self, val):
        return np.reciprocal(np.power(np.sqrt(2.0 * np.pi), val))

    # Probability of y: P(y | x, i, theta) is calculated according to equation (2.3c)
    def calc_P_y(self, x, y, i):
        
        diff = np.square(y - self.linear_regression(x, i))
        ep = np.exp(-0.5 * self.calc_recip_var(self.var[i], 2) * diff)
        sd = self.calc_recip_var(self.var[i], self.D)
        pi = self.calc_recip_pi(self.D)

        return pi * sd * ep

    
    def calc_det(self, cov):
        
        vals, vecs = np.linalg.eigh(cov)
        det = 1.0
        for v in vals:
            det *= v
        
        return det
    

    # Probability of xyi: P(x, y, i | theta) is calculated according to equation (2.2)
    def calc_P_xyi(self, x, y, i):

        diff = diff = np.square(y - self.linear_regression(x, i))
        maha = calc_maha(self, x, self.mu[i], self.Sigma[i])
        ep = -0.5 * (maha + self.calc_recip_var(self.var[i], 2) * diff)
        pi = self.calc_recip_pi(self.D + self.N)
        sd = self.calc_recip_var(self.var[i], self.D)
        invdet = np.reciprocal(sqrt(calc_det(self.Sigma[i])))

        return (-1/self.M) * pi * sd * ep
    
        
    def E_step(self, x, y):

        s = 0
        for i, p in enumerate(self.P_xyi):
            self.P_xyi[i] = self.calc_P_xyi(x, y, i)
            s += self.P_xyi[i]

        for i, p in enumerate(self.P_xyi):
            self.posterior_i[i] = self.P_xyi[i] / s

        return self.posterior_i

    def M_step(self, x, y):
        pass
        

if __name__ == '__main__':

    ngnet = NGnet(4, 3, 3)

    print(ngnet.obtain_output(2 * np.random.rand(4, 1) - 1))
    
    # print(ngnet.evaluate_Normalized_Gaussian_value(2 * np.random.rand(3, 1) - 1, 0))
    
    # for mu, Sigma, W in zip(ngnet.mu, ngnet.Sigma, ngnet.W):
    #     print(mu, Sigma, W)
        
