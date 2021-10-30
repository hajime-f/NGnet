import numpy as np
import numpy.linalg as LA
import random
import pdb

# This code is the implementation of the Normalized Gaussian Network (NGnet) with
# online EM algorithm.

# In the details, see the article shown below.

# Masa-aki Sato & Shin Ishii
# On-line EM Algorithm for the Normalized Gaussian Network
# Neural Computation, 2000 Vol.12, pp.407-432, 2000
# https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.37.3704&rep=rep1&type=pdf

class NGnet_OEM:

    mu = []        # Center vectors of N-dimensional Gaussian functions
    Sigma = []     # Covariance matrices of N-dimensional Gaussian functions
    Sigma_inv = [] # Inverse matrices of the above covariance matrices of N-dimensional Gaussian functions
    Lambda = []    # Auxiliary variable to calculate covariance matrix Sigma
    W = []         # Linear regression matrices in units
    var = []       # Variance of D-dimensional Gaussian functions
    
    N = 0  # Dimension of input data
    D = 0  # Dimension of output data
    M = 0  # Number of units

    one = []
    x = []
    y2 = []
    xy = []

    eta = []
    lam = 0
    alpha = 0
    Nlog2pi = 0
    Dlog2pi = 0

    posterior_i = []   # Posterior probability that the i-th unit is selected for each observation

    def __init__(self, N, D, M, lam=0.998, alpha=0.1):

        self.mu = [2 * np.random.rand(N, 1) - 1 for i in range(M)]

        for i in range(M):
            x = [random.random() for i in range(N)]
            self.Sigma.append(np.diag(x))
        
        for i in range(M):
            w = 2 * np.random.rand(D, N) - 1
            w_tilde = np.insert(w, np.shape(w)[1], 1.0, axis=1)
            self.W.append(w_tilde)

        self.var = [np.array(0.5) for i in range(M)]

        self.eta = 1 / ((1 + lam) / 0.9999)

        self.one = [np.array(0.01) for i in range(M)]
        self.x = [np.ones((N, 1)) for i in range(M)]
        self.y2 = [np.array(1.0) for i in range(M)]
        self.xy = [np.ones((N+1, D)) for i in range(M)]
        
        self.N = N
        self.D = D
        self.M = M
        self.lam = lam
        self.alpha = alpha
        self.Nlog2pi = N * np.log(2 * np.pi)
        self.Dlog2pi = D * np.log(2 * np.pi)

        
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
            sum_g_j += self.multinorm_pdf(x, self.mu[j], self.Sigma_inv[j])

        # Numerator of equation (2.1b)
        g_i = self.multinorm_pdf(x, self.mu[i], self.Sigma_inv[i])

        # Equation (2.1b)
        N_i = g_i / sum_g_j
        
        return N_i


    # This function calculates multivariate Gaussian G(x) according to equation (2.1c)
    def multinorm_pdf(self, x, mean, covinv):
        
        logdet = np.log(LA.det(covinv))
        diff = x - mean

        logpdf = -0.5 * (self.Nlog2pi - logdet + (diff.T @ covinv @ diff))
        
        if np.isnan(logdet):
            pdb.set_trace()
        
        return np.exp(logpdf)    
    

    def generate_x_tilde(self, x_t):
        
        tmp = [x_t[j].item() for j in range(len(x_t))]
        tmp.append(1)
        x_tilde = np.array(tmp).reshape(-1, 1)
        
        return x_tilde
    
    
    # This function calculates W_i * x.
    def linear_regression(self, x_t, i):

        x_tilde = self.generate_x_tilde(x_t)

        return self.W[i] @ x_tilde
    

    def batch_learning(self, x_list, y_list):
        
        if len(x_list) != len(y_list):
            print('Error: The number of input vectors x is not equal to the number of output vectors y.')
            exit()
        else:
            self.T = len(x_list)

        self.posterior_i = []
        self.batch_E_step(x_list, y_list)
        self.batch_M_step(x_list, y_list)
        
    
    # This function executes E-step written by equation (3.1)
    def batch_E_step(self, x_list, y_list):

        for x_t, y_t in zip(x_list, y_list):
            sum_p = 0
            for i in range(self.M):
                sum_p += self.batch_calc_P_xyi(x_t, y_t, i)
            p_t = []
            for i in range(self.M):
                p = self.batch_calc_P_xyi(x_t, y_t, i)
                p_t.append(p / sum_p)
            self.posterior_i.append(p_t)


    # This function calculates equation (2.2)
    def batch_calc_P_xyi(self, x_t, y_t, i):

        # Equation (2.3a)
        P_i = 1 / self.M
        
        # Equation (2.3b)
        P_x = self.batch_multinorm_pdf(x_t, self.mu[i], self.Sigma[i])
        
        # Equation (2.3c)
        diff = y_t - self.linear_regression(x_t, i)
        P_y = self.norm_pdf(diff, self.var[i])

        # Equation (2.2)
        P_xyi = P_i * P_x * P_y   # Joint Probability of i, x and y
        
        return P_xyi


    # This function calculates equation (2.1c)
    def batch_multinorm_pdf(self, x, mean, cov):

        alpha_I = np.diag([0.00001 for _ in range(self.N)])
        cov = cov + alpha_I
        
        logdet = np.log(LA.det(cov))
        covinv = LA.inv(cov)
        diff = x - mean

        logpdf = -0.5 * (self.Nlog2pi + logdet + (diff.T @ covinv @ diff))
        
        return np.exp(logpdf)
    

    # This function executes M-step written by equation (3.4)
    def batch_M_step(self, x_list, y_list):
        
        self.batch_Sigma_update(x_list)
        self.batch_mu_update(x_list)
        self.batch_W_update(x_list, y_list)
        self.batch_var_update(x_list, y_list)
            

    # This function updates mu according to equation (3.4a)
    def batch_mu_update(self, x_list):

        for i in range(self.M):
            sum_1 = 0
            sum_mu = 0
            for t, x_t in enumerate(x_list):
                sum_1 += self.posterior_i[t][i]
                sum_mu += x_t.T * self.posterior_i[t][i]
            self.mu[i] = (sum_mu / sum_1).T


    # This function updates Sigma according to equation (3.4b)
    def batch_Sigma_update(self, x_list):

        for i in range(self.M):
            sum_1 = 0
            sum_diff = 0
            for t, x_t in enumerate(x_list):
                sum_1 += self.posterior_i[t][i]
                diff = x_t - self.mu[i]
                sum_diff += (diff @ diff.T) * self.posterior_i[t][i]
            self.Sigma[i] = sum_diff / sum_1
            

    # This function updates W according to equation (3.4c)
    def batch_W_update(self, x_list, y_list):

        alpha_I = np.diag([0.00001 for i in range(self.N+1)])   # Regularization matrix
        for i, W_i in enumerate(self.W):
            sum_xx = 0
            sum_yx = 0
            for t, (x_t, y_t) in enumerate(zip(x_list, y_list)):
                x_tilde = np.insert(x_t, len(x_t), 1.0).reshape(-1, 1)
                sum_xx += (x_tilde * x_tilde.T * self.posterior_i[t][i]) / self.T
                sum_yx += (y_t * x_tilde.T * self.posterior_i[t][i]) / self.T
            self.W[i] = sum_yx @ LA.inv(sum_xx + alpha_I)


    # This function updates var according to equation (3.4d)
    def batch_var_update(self, x_list, y_list):
        
        for i, var_i in enumerate(self.var):
            sum_1 = 0
            sum_diff = 0
            for t, (x_t, y_t) in enumerate(zip(x_list, y_list)):
                sum_1 += self.posterior_i[t][i]
                diff = y_t - self.linear_regression(x_t, i).T
                sum_diff += (diff @ diff.T) * self.posterior_i[t][i]
            self.var[i] = (1/self.D) * (sum_diff / sum_1)


    # This function calculates the log likelihood according to equation (3.3)
    def batch_calc_log_likelihood(self, x_list, y_list):

        log_likelihood = 0
        for x_t, y_t in zip(x_list, y_list):
            p_t = 0
            for i in range(self.M):
                p_t += self.batch_calc_P_xyi(x_t, y_t, i)
            log_likelihood += np.log(p_t)
            
        return log_likelihood.item()
    
    
    def online_learning(self, x_t, y_t):

        self.posterior_i = []
        self.E_step(x_t, y_t)
        self.M_step(x_t, y_t)


    def set_Sigma_Lambda(self):

        alpha_I = np.diag([0.00001 for _ in range(self.N)])
        for i in range(self.M):
            self.Sigma_inv.append(LA.inv(self.Sigma[i] + alpha_I))

        for i in range(self.M):
            u1 = - self.Sigma_inv[i] @ self.mu[i]
            u2 = - self.mu[i].T @ self.Sigma_inv[i]
            u3 = 1 + self.mu[i].T @ self.Sigma_inv[i] @ self.mu[i]
            t1 = np.concatenate([self.Sigma_inv[i], u1], 1)
            t2 = np.concatenate([u2, u3], 1)
            self.Lambda.append(np.concatenate([t1, t2], 0))
        
        
    # This function executes E-step written by equation (3.1)
    def E_step(self, x_t, y_t):
        
        p = []
        for i in range(self.M):
            p.append(self.calc_P_xyi(x_t, y_t, i).item())
        p_sum = sum(p)

        for i in range(self.M):
            self.posterior_i.append(p[i] / p_sum)


    # This function calculates equation (2.2)
    def calc_P_xyi(self, x_t, y_t, i):

        # Equation (2.3a)
        P_i = 1 / self.M
        
        # Equation (2.3b)
        P_x = self.multinorm_pdf(x_t, self.mu[i], self.Sigma_inv[i])
        
        # Equation (2.3c)
        diff = y_t - self.linear_regression(x_t, i)
        P_y = self.norm_pdf(diff, self.var[i])

        # Equation (2.2)
        P_xyi = P_i * P_x * P_y   # Joint Probability of i, x and y
        
        return P_xyi


    # This function calculates normal function according to equation (2.3c)
    def norm_pdf(self, diff, var):
        
        log_pdf1 = self.D * np.log(var)
        log_pdf2 = (1 / var) * (diff.T @ diff)

        return np.exp(-0.5 * (self.Dlog2pi + log_pdf1 + log_pdf2))
        
    
    def M_step(self, x_t, y_t):
        
        self.update_weighted_mean(x_t, y_t)
        self.update_mu()
        self.update_Lambda(x_t)
        self.regularization()
        self.update_Sigma_inv()
        self.update_W(x_t, y_t)
        # self.update_var()


    # This function updates weighted means according to equation (4.2)
    def update_weighted_mean(self, x_t, y_t):

        # self.eta = (1 + self.lam) / self.eta
        self.eta = 0.5
        x_tilde = self.generate_x_tilde(x_t)

        for i in range(self.M):
            self.one[i] = self.one[i] + self.eta * (self.posterior_i[i] - self.one[i])   # scalar <<1>>
            self.x[i] = self.x[i] + self.eta * (x_t * self.posterior_i[i] - self.x[i])   # (N x 1)-dimensional vector <<x>>
            self.y2[i] = self.y2[i] + self.eta * (y_t.T @ y_t * self.posterior_i[i] - self.y2[i])  # scalar <<y^2>>
            self.xy[i] = self.xy[i] + self.eta * (x_tilde @ y_t.T * self.posterior_i[i] - self.xy[i])  # ((N+1) x D)-dimensional matrix <<xy>>


    # This function updates mu according to equation (4.5a)
    def update_mu(self):
        
        for i in range(self.M):
            self.mu[i] = self.x[i] / self.one[i]
            

    # This function updates Lambda according to equation (4.6a)
    def update_Lambda(self, x_t):
        
        x_tilde = self.generate_x_tilde(x_t)        

        for i in range(self.M):
            
            t1 = self.Lambda[i] @ x_tilde
            t2 = x_tilde.T @ self.Lambda[i]
            numerator = self.posterior_i[i] * (t1 @ t2)   # ((N+1) x (N+1))-dimensional matrix
            denominator = (1 / self.eta) - 1 + self.posterior_i[i] * (t2 @ x_tilde)   # scalar
            
            self.Lambda[i] = (1 / (1 - self.eta)) * (self.Lambda[i] - numerator / denominator)
            

    # This function regularizes Lambda according to equation (5.12)
    def regularization(self):

        tmp = [self.alpha for j in range(self.N)]
        tmp.append(1)
        nu = np.array(tmp).reshape(-1, 1)
        
        for i in range(self.M):

            t1 = self.Lambda[i] @ nu
            t2 = nu.T @ self.Lambda[i]
            t3 = self.eta * self.posterior_i[i]
            k = t3 / (1 + t3 * t2 @ nu)
            self.Lambda[i] = self.Lambda[i] - k * t1 @ t2


    # This function picks up the inverse of Sigma from Lambda according to equation (4.7)
    def update_Sigma_inv(self):
        
        for i in range(self.M):
            self.Sigma_inv[i] = self.Lambda[i][0:self.N, 0:self.N] * self.one[i]
        

    # This function updates W according to equation (4.8)
    def update_W(self, x_t, y_t):

        x_tilde = self.generate_x_tilde(x_t)
        
        for i in range(self.M):
            
            diff = y_t.reshape(-1, 1) - self.linear_regression(x_t, i)
            self.W[i] = self.W[i] + self.eta * self.posterior_i[i] * (diff @ x_tilde.T @ self.Lambda[i])
        

    # This function updates variance according to equation (4.5d)
    def update_var(self):

        for i in range(self.M):
            self.var[i] = (self.y2[i] - np.trace(self.W[i] @ self.xy[i])) / (self.one[i] * self.D)
            if self.var[i] < 0:
                pdb.set_trace()

    
    # This function calculates the log likelihood according to equation (3.3)
    def calc_log_likelihood(self, x_t, y_t):

        p_t = 0
        for i in range(self.M):
            p_t += self.calc_P_xyi(x_t, y_t, i)
        log_likelihood = np.log(p_t)
        
        return log_likelihood.item()
    
    
def func1(x_1, x_2):
    s = np.sqrt(np.power(x_1, 2) + np.power(x_2, 2))
    y = np.sin(s) / s
    return np.array(y).reshape(-1, 1)

def func2(x_1, x_2, x_3):
    s = np.sqrt(np.power(x_1, 2) + np.power(x_2, 2))
    y = [np.sin(s).item() / s.item(), (1.0 - np.sin(s)).item() / s.item()]
    return np.array(y).reshape(-1, 1)


if __name__ == '__main__':

    N = 2
    D = 1
    M = 15

    pt_T = 300
    learning_T = 1000
    inference_T = 1000
    
    ngnet = NGnet_OEM(N, D, M)

    # Preparing for pre-training data
    pt_x_list = [20 * np.random.rand(N, 1) - 10 for _ in range(pt_T)]
    pt_y_list = [func1(x_t[0], x_t[1]) for x_t in pt_x_list]
    
    pdb.set_trace()
    
    
    # Training NGnet to initialize parameters
    previous_likelihood = -10 ** 6
    next_likelihood = -10 ** 5
    while abs(next_likelihood - previous_likelihood) > 5:
        ngnet.batch_learning(pt_x_list, pt_y_list)
        previous_likelihood = next_likelihood
        next_likelihood = ngnet.batch_calc_log_likelihood(pt_x_list, pt_y_list)
        print(next_likelihood)
        if previous_likelihood >= next_likelihood:
            print('*** Warning: likelihood decreases!')

    ngnet.set_Sigma_Lambda()

    # Preparing for learning data
    learning_x_list = [20 * np.random.rand(N, 1) - 10 for _ in range(learning_T)]
    learning_y_list = [func1(x_t[0], x_t[1]) for x_t in learning_x_list]
    
    # Online training NGnet
    for x_t, y_t in zip(learning_x_list, learning_y_list):
        ngnet.online_learning(x_t, y_t)
    
    exit()
    
    
    # Preparing for evaluation data to evaluate the log likelihood
    eval_x_list = []
    for t in range(eval_T):
        eval_x_list.append(20 * np.random.rand(N, 1) - 10)
    eval_y_list = []
    for x_t in eval_x_list:
        y = func1(x_t[0], x_t[1])
        eval_y_list.append(y)
    ngnet.set_eval_data(eval_x_list, eval_y_list)
    
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


        
