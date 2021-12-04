from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import sys, math, random, pdb

class NGnet_OEM:
    
    mu = []           # Center vectors of N-dimensional Gaussian functions
    Sigma = []        # Covariance matrices of N-dimensional Gaussian functions
    Sigma_inv = []    # Inverse matrices of the above covariance matrices of N-dimensional Gaussian functions
    Lambda = []       # Auxiliary variable to calculate covariance matrix Sigma
    W = []            # Linear regression matrices in units
    
    var = []          # Variance of D-dimensional Gaussian functions
    posterior_i = []  # Posterior probability that the i-th unit is selected for each observation

    N = 0       # Dimension of input data
    D = 0       # Dimension of output data
    M = 0       # Number of units
    
    T = 0       # Number of learning data

    one = []
    x = []
    y2 = []
    xy = []
    
    eta = 0.5
    lam = 0
    alpha = 0
    Nlog2pi = 0  # N * log(2 * pi)
    Dlog2pi = 0  # D * log(2 * pi)
    
    def __init__(self, N, D, M, T, lam=0.998, alpha=0.1):

        for i in range(M):
            self.mu.append(2 * np.random.rand(N, 1) - 1)
        
        for i in range(M):
            x = [random.random() for i in range(N)]
            self.Sigma.append(np.diag(x))

        for i in range(M):
            w = 2 * np.random.rand(D, N) - 1
            w_tilde = np.insert(w, np.shape(w)[1], 1.0, axis=1)
            self.W.append(w_tilde)

        for i in range(M):
            self.var.append(1)

        self.eta = 0.5 / (0.5 + lam)

        self.one = [np.array(1) for i in range(M)]
        self.x = [np.zeros((N, 1)) for i in range(M)]
        self.y2 = [np.array(1.0) for i in range(M)]
        self.xy = [np.ones((N+1, D)) for i in range(M)]
        
        self.N = N
        self.D = D
        self.M = M
        
        self.T = T
        
        Nlog2pi = N * np.log(2 * np.pi)
        Dlog2pi = D * np.log(2 * np.pi)
            

    ### The functions written below are to calculate the output y given the input x
        
    # This function returns the output y corresponding the input x according to equation (2.1a)
    def get_output_y(self, x):

        # Initialization of the output vector y
        y = np.array([0.0] * self.D).reshape(-1, 1)

        # Equation (2.1a)
        for i in range(self.M):
            N_i = self.evaluate_Normalized_Gaussian_value(x, i)  # N_i(x)
            Wx = self.linear_regression(x, i)  # W_i * x
            y += N_i * Wx
        
        return y


    # This function calculates N_i(x) according to equation (2.1b)
    def evaluate_Normalized_Gaussian_value(self, x, i):

        # Denominator of equation (2.1b)
        sum_g_j = 0
        for j in range(self.M):
            sum_g_j += self.batch_multinorm_pdf(x, self.mu[j], self.Sigma[j])

        # Numerator of equation (2.1b)
        g_i = self.batch_multinorm_pdf(x, self.mu[i], self.Sigma[i])

        # Equation (2.1b)
        N_i = g_i / sum_g_j
        
        return N_i


    # This function calculates multivariate Gaussian G(x) according to equation (2.1c)
    def batch_multinorm_pdf(self, x, mean, cov):

        cov_alpha = cov + np.diag([0.00001 for i in range(self.N)])
        
        logdet = np.log(LA.det(cov_alpha))
        covinv = LA.inv(cov_alpha)
        
        diff = x - mean

        logpdf = -0.5 * (self.Nlog2pi + logdet + (diff.T @ covinv @ diff))

        return np.exp(logpdf)


    def generate_x_tilde(self, x_t):
        
        tmp = [x_t[j].item() for j in range(len(x_t))]
        tmp.append(1)
        x_tilde = np.array(tmp).reshape(-1, 1)
        
        return x_tilde
    
    
    # This function calculates W_i * x.
    def linear_regression(self, x, i):

        x_tilde = self.generate_x_tilde(x)
        Wx = np.dot(self.W[i], x_tilde)

        return Wx
    
    
    ### The functions written below are to learn the parameters according to the batch EM algorithm.
    
    def batch_learning(self, x_list, y_list):
        
        if len(x_list) != len(y_list):
            print('Error: The number of input vectors x is not equal to the number of output vectors y.')
            exit()
            
        self.posterior_i = []
        self.batch_E_step(x_list, y_list)
        self.batch_M_step(x_list, y_list)


    # This function executes E-step written by equation (3.1)
    def batch_E_step(self, x_list, y_list):

        for x_t, y_t in zip(x_list, y_list):
            sum_p = 0
            for i in range(self.M):
                p = self.batch_calc_P_xyi(x_t, y_t, i)
                if p == 0:
                    p = sys.float_info.epsilon
                sum_p += p
            p_t = []
            for i in range(self.M):
                p = self.batch_calc_P_xyi(x_t, y_t, i)
                p_t.append(p / sum_p)
            self.posterior_i.append(p_t)


    # This function calculates equation (2.2)
    def batch_calc_P_xyi(self, x, y, i):

        # Equation (2.3a)
        P_i = 1 / self.M

        # Equation (2.3b)
        P_x = self.batch_multinorm_pdf(x, self.mu[i], self.Sigma[i]).item()
        
        # Equation (2.3c)
        diff = y.reshape(-1, 1) - self.linear_regression(x, i)
        P_y = self.norm_pdf(diff, self.var[i]).item()
        
        # Equation (2.2)
        P_xyi = P_i * P_x * P_y

        return P_xyi

    
    # This function calculates normal function according to equation (2.3c)
    def norm_pdf(self, diff, var):
        
        log_pdf1 = - self.Dlog2pi / 2
        log_pdf2 = - self.D/2 * np.log(var)
        log_pdf3 = - (1/(2 * var)) * (diff.T @ diff)
        return np.exp(log_pdf1 + log_pdf2 + log_pdf3)

    
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
                diff = y_t.reshape(-1, 1) - self.linear_regression(x_t, i)
                sum_diff += (diff.T @ diff) * self.posterior_i[t][i]
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
    

    ### The functions written below are to learn the parameters according to the online EM algorithm.
            
    def online_learning(self, x_t, y_t):

        # pdb.set_trace()
        
        self.posterior_i = []
        self.online_E_step(x_t, y_t)
        self.online_M_step(x_t, y_t)
    

    # This function executes E-step written by equation (3.1)
    def online_E_step(self, x_t, y_t):
        
        p = []
        for i in range(self.M):
            tmp = self.online_calc_P_xyi(x_t, y_t, i)
            if tmp == 0:
                tmp = sys.float_info.epsilon
            p.append(tmp)
        p_sum = sum(p)

        for i in range(self.M):
            self.posterior_i.append(p[i] / p_sum)


    # This function calculates equation (2.2)
    def online_calc_P_xyi(self, x_t, y_t, i):

        # Equation (2.3a)
        P_i = 1 / self.M
        
        # Equation (2.3b)
        P_x = self.online_multinorm_pdf(x_t, self.mu[i], self.Sigma_inv[i])
        
        # Equation (2.3c)
        diff = y_t - self.linear_regression(x_t, i)
        P_y = self.norm_pdf(diff, self.var[i])

        # Equation (2.2)
        P_xyi = P_i * P_x * P_y   # Joint Probability of i, x and y
        
        return P_xyi


    # This function calculates multivariate Gaussian G(x) according to equation (2.1c)
    def online_multinorm_pdf(self, x, mean, covinv):

        logdet = np.log(LA.det(covinv))
        diff = x - mean

        logpdf = -0.5 * (self.Nlog2pi - logdet + (diff.T @ covinv @ diff))
        
        return np.exp(logpdf)    


    def online_M_step(self, x_t, y_t):

        pdb.set_trace()
        
        self.update_weighted_mean(x_t, y_t)
        self.online_mu_update()
        self.online_Lambda_update(x_t)
        self.online_Sigma_inv_update()


    # This function updates weighted means according to equation (4.2)
    def update_weighted_mean(self, x_t, y_t):

        # self.eta = self.eta / (self.eta + self.lam)
        self.eta = 0.9
        
        x_tilde = self.generate_x_tilde(x_t)

        for i in range(self.M):
            self.one[i] = self.one[i] + self.eta * (self.posterior_i[i] - self.one[i])   # scalar <<1>>
            self.x[i] = self.x[i] + self.eta * (x_t * self.posterior_i[i] - self.x[i])   # (N x 1)-dimensional vector <<x>>
            self.y2[i] = self.y2[i] + self.eta * (y_t.T @ y_t * self.posterior_i[i] - self.y2[i])  # scalar <<y^2>>
            self.xy[i] = self.xy[i] + self.eta * (x_tilde @ y_t.T * self.posterior_i[i] - self.xy[i])  # ((N+1) x D)-dimensional matrix <<xy>>


    # This function updates mu according to equation (4.5a)
    def online_mu_update(self):
        
        for i in range(self.M):
            self.mu[i] = self.x[i] / self.one[i]
            

    # This function updates Lambda according to equation (4.6a)
    def online_Lambda_update(self, x_t):
        
        x_tilde = self.generate_x_tilde(x_t)        

        for i in range(self.M):
            
            t1 = self.Lambda[i] @ x_tilde
            t2 = x_tilde.T @ self.Lambda[i]
            numerator = self.posterior_i[i] * (t1 @ t2)   # ((N+1) x (N+1))-dimensional matrix
            denominator = ((1 / self.eta) - 1) + self.posterior_i[i] * (t2 @ x_tilde)   # scalar

            self.Lambda[i] = (1 / (1 - self.eta)) * (self.Lambda[i] - numerator / denominator)


    # This function picks up the inverse of Sigma from Lambda according to equation (4.7)
    def online_Sigma_inv_update(self):
        
        for i in range(self.M):
            lambda_tmp = self.Lambda[i] * self.one[i]
            self.Sigma_inv[i] = lambda_tmp[0:self.N, 0:self.N]
            



    
    
def func(x1, x2):

    s = np.sqrt(np.power(x1, 2) + np.power(x2, 2))
    return np.sin(s) / s

def func2(x1, x2, x3):

    y1 = np.sin(x1) + np.cos(x2)
    y2 = np.sin(x2) - np.cos(x3)

    y = np.array([y1.item(), y2.item()])
    
    return y.reshape(-1, 1)

if __name__ == '__main__':
    
    N = 2
    D = 1
    M = 15
    learning_T = 1000
    inference_T = 1000
    
    ngnet = NGnet_OEM(N, D, M, learning_T)
    
    # Preparing for learning data
    learning_x_list = [20 * np.random.rand(N, 1) - 10 for _ in range(learning_T)]
    learning_y_list = [func(x_t[0], x_t[1]) for x_t in learning_x_list]
    # learning_y_list = [func(x_t[0], x_t[1], x_t[2]) for x_t in learning_x_list]
    
    # Training NGnet
    previous_likelihood = -10 ** 6
    next_likelihood = -10 ** 5
    while abs(next_likelihood - previous_likelihood) > 5:
        ngnet.batch_learning(learning_x_list, learning_y_list)
        previous_likelihood = next_likelihood
        next_likelihood = ngnet.batch_calc_log_likelihood(learning_x_list, learning_y_list)
        print(next_likelihood)
        if previous_likelihood >= next_likelihood:
            print('Warning: Next likelihood is smaller than previous.')

    ngnet.set_Sigma_Lambda()

    # Preparing for learning data
    learning_x_list = [20 * np.random.rand(N, 1) - 10 for _ in range(learning_T)]
    learning_y_list = [func(x_t[0], x_t[1]) for x_t in learning_x_list]
    
    # Online training NGnet
    for x_t, y_t in zip(learning_x_list, learning_y_list):
        ngnet.online_learning(x_t, y_t)
        
    pdb.set_trace()
    
    # Inference the output y
    inference_x_list = []
    for t in range(inference_T):
        inference_x_list.append(20 * np.random.rand(N, 1) - 10)
    inference_y_list = []
    for x_t in inference_x_list:
        inference_y_list.append(ngnet.get_output_y(x_t))

    # Plot graph
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

    x1_real = np.arange(-10.0, 10.0, 0.02)
    x2_real = np.arange(-10.0, 10.0, 0.02)

    X1_real, X2_real = np.meshgrid(x1_real, x2_real)
    Y1_real = func(X1_real, X2_real)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("Y")

    ax.plot_wireframe(X1_real, X2_real, Y1_real)
    ax.plot(X1_appx, X2_appx, Y1_appx, marker="o", linestyle='None', color="r")

    plt.show()
    
    
