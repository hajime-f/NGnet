from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
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

class NGnet:
    
    mu = []     # Center vectors of N-dimensional Gaussian functions
    Sigma = []  # Covariance matrices of N-dimensional Gaussian functions
    W = []      # Linear regression matrices in the units
    
    N  = 0      # The dimension of input data
    D = 0       # The dimension of output data
    M = 0       # The number of units

    var = []    # Covariance matrices of D-dimensional Gaussian functions
    posterior_i = []   # Posterior probability that the i-th unit is selected for each observation

    offline_iteration_num = 0
    
    def __init__(self, N, D, M):
        
        # The constructor initializes mu, Sigma and W.
        for i in range(M):
            self.mu.append(2 * np.random.rand(N, 1) - 1)
        
        for i in range(M):
            x = [random.random() for i in range(N)]
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
            self.var.append(1)
        

    ### The functions written below are to calculate the output y given the input x
        
    # This function returns the output y corresponding the input x.
    def get_output_y(self, x):

        # Initialization of the output vector y
        y = np.array([0.0] * self.D).reshape(-1, 1)

        # Equation (2.1a)
        for i in range(self.M):
            N_i = self.evaluate_Normalized_Gaussian_value(x, i)  # N_i(x)
            Wx = self.linear_regression(x, i)  # W_i * x
            y += N_i * Wx
        
        return y


    # This function calculates N_i(x).
    def evaluate_Normalized_Gaussian_value(self, x, i):

        # Denominator of equation (2.1b)
        sum_g_j = 0
        for j in range(self.M):
            sum_g_j += self.multinorm_pdf2(x, self.mu[j], self.Sigma[j])

        # Numerator of equation (2.1b)
        g_i = self.multinorm_pdf2(x, self.mu[i], self.Sigma[i])

        # Equation (2.1b)
        N_i = g_i / sum_g_j
        
        return N_i        
    
    
    def multinorm_pdf(self, x, mean, cov):
        
        # "eigh" assumes the matrix is Hermitian.
        vals, vecs = LA.eigh(cov)
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
        
        return np.exp(-0.5 * (rank * log2pi + maha + logdet))


    def multinorm_pdf2(self, x, mean, cov):

        Nlog2pi = self.N * np.log(2 * np.pi)
        logdet = np.log(LA.det(cov))
        covinv = LA.inv(cov)
        diff = x - mean

        logpdf = -0.5 * (Nlog2pi + logdet + (diff.T @ covinv @ diff))

        return np.exp(logpdf)
    
    
    # This function calculates W_i * x.
    def linear_regression(self, x, i):

        x_tilde = np.insert(x, len(x), 1.0).reshape(-1, 1)
        Wx = np.dot(self.W[i], x_tilde)

        return Wx

    
    ### The functions written below are to learn the parameters according to the EM algorithm.

    def offline_learning(self, x_list, y_list):
        
        if len(x_list) != len(y_list):
            print('Error: The number of input vectors x is not equal to the number of output vectors y.')
            exit()

        self.posterior_i = []
        self.offline_E_step(x_list, y_list)
        self.offline_M_step(x_list, y_list)


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

            
    # This function calculates equation (2.2)
    def calc_P_xyi(self, x, y, i):

        # Equation (2.3a)
        P_i = 1 / self.M

        # Equation (2.3b)
        P_x = self.multinorm_pdf2(x, self.mu[i], self.Sigma[i])

        # Equation (2.3c)
        diff = y.reshape(-1, 1) - self.linear_regression(x, i)
        P_y = self.norm_pdf(diff, self.var[i])

        # Equation (2.2)
        P_xyi = P_i * P_x * P_y

        return P_xyi


    def norm_pdf(self, diff, var):
        
        log_pdf1 = - self.D/2 * np.log(2 * np.pi)
        log_pdf2 = - self.D * np.log(var)
        log_pdf3 = - (1/(2 * np.power(var, 2))) * (diff.T @ diff)
        return np.exp(log_pdf1 + log_pdf2 + log_pdf3)
    
            
    # This function executes M-step written by equation (3.4)
    def offline_M_step(self, x_list, y_list):
        
        self.offline_Sigma_update(x_list)
        self.offline_mu_update(x_list)
        self.offline_W_update(x_list, y_list)
        self.offline_var_update(x_list, y_list)
    
        
    # This function updates mu according to equation (3.4a)
    def offline_mu_update(self, x_list):

        for i, mu_i in enumerate(self.mu):
            sum_1 = 0
            sum_mu = 0
            for t, x_t in enumerate(x_list):
                sum_1 += self.posterior_i[t][i]
                sum_mu += x_t.T * self.posterior_i[t][i]
            self.mu[i] = (sum_mu / sum_1).T


    # This function updates Sigma according to equation (3.4b)
    def offline_Sigma_update(self, x_list):

        for i, Sigma_i in enumerate(self.Sigma):
            sum_1 = 0
            sum_diff = 0
            for t, x_t in enumerate(x_list):
                sum_1 += self.posterior_i[t][i]
                diff = x_t - self.mu[i]
                sum_diff += (diff * diff.T) * self.posterior_i[t][i]
            self.Sigma[i] = sum_diff / sum_1
            

    # This function updates W according to equation (3.4c)
    def offline_W_update(self, x_list, y_list):

        for i, W_i in enumerate(self.W):
            sum_xx = 0
            sum_yx = 0
            alpha_I = np.diag([0.00001 for i in range(self.N+1)])   # Regularization matrix
            for t, (x_t, y_t) in enumerate(zip(x_list, y_list)):
                x_tilde = np.insert(x_t, len(x_t), 1.0).reshape(-1, 1)
                sum_xx = x_tilde * x_tilde.T * self.posterior_i[t][i]
                sum_yx = y_t * x_tilde.T * self.posterior_i[t][i]
            self.W[i] = np.dot(sum_yx, LA.inv(sum_xx + alpha_I))


    # This function updates var according to equation (3.4d)
    def offline_var_update(self, x_list, y_list):
        
        for i, var_i in enumerate(self.var):
            sum_1 = 0
            sum_diff = 0
            for t, (x_t, y_t) in enumerate(zip(x_list, y_list)):
                sum_1 += self.posterior_i[t][i]
                diff = y_t - self.linear_regression(x_t, i)
                sum_diff += (diff.T @ diff) * self.posterior_i[t][i]
            self.var[i] = np.sqrt((1/self.D) * (sum_diff / sum_1))


    # This function calculates the log likelihood according to equation (3.3)
    def calc_log_likelihood(self, x_list, y_list):

        self.offline_iteration_num += 1

        log_likelihood = 0
        for x_t, y_t in zip(x_list, y_list):
            p_t = 0
            for i in range(self.M):
                p = self.calc_P_xyi(x_t, y_t, i)
                p_t += p
            log_likelihood += np.log(p_t)

        return log_likelihood


# Mexican hat function
def func1(x_1, x_2):
    s = np.sqrt(np.power(x_1, 2) + np.power(x_2, 2))
    return np.sin(s) / s


if __name__ == '__main__':

    N = 2
    D = 1
    M = 10
    
    ngnet = NGnet(N, D, M)

    x_list = []
    y_list = []
    T = 1000

    # Preparing for learning data
    for t in range(T):
        x_list.append(20 * np.random.rand(N, 1) - 10)
    for x_t in x_list:
        y_list.append(np.array(func1(x_t[0], x_t[1])))

    # Training NGnet
    for i in range(20):
        ngnet.offline_learning(x_list, y_list)
        print(ngnet.calc_log_likelihood(x_list, y_list))

    # Inference the output y
    appx_list = []
    for x_t in x_list:
        y_t = ngnet.get_output_y(x_t)
        appx_list.append(y_t)
        
    # Plot graph
    x1_list = []
    x2_list = []
    y1_list = []
    for x_t, y_t in zip(x_list, y_list):
        x1_list.append(x_t[0].item())
        x2_list.append(x_t[1].item())
        y1_list.append(y_t.item())
    X1_appx = np.array(x1_list)
    X2_appx = np.array(x2_list)
    Y1_appx = np.array(y1_list)

    x1_real = np.arange(-10.0, 10.0, 0.02)
    x2_real = np.arange(-10.0, 10.0, 0.02)

    X1_real, X2_real = np.meshgrid(x1_real, x2_real)
    Y1_real = func1(X1_real, X2_real)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("Y")

    ax.plot_wireframe(X1_real, X2_real, Y1_real)
    ax.plot(X1_appx, X2_appx, Y1_appx, marker="o", linestyle='None', color="r")

    plt.show()
    

    
