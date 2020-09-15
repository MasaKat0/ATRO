import numpy as np
from scipy.optimize import minimize


class SVM():
    def __init__(self, C):
        self.C = C
        
    def fit(self, x, y):
        N, dim = x.shape
        beta0 = np.zeros(dim+N)
        
        fun = lambda beta: self.objective_function(beta, N)
        
        cons = ({'type': 'ineq', 'fun': lambda beta:  self.contraint_function(beta, x, y)},
                {'type': 'ineq', 'fun': lambda beta: beta[:N]})
        
        res = minimize(fun, beta0, method='SLSQP', constraints=cons)
        
        self.xi, self.gamma = res.x[:N], res.x[N:]
                
    def objective_function(self, beta, N):
        xi, omega = beta[:N], beta[N:]
        
        return np.sum(xi) + (self.C/2)*np.sqrt(np.mean(omega**2))
        
    def contraint_function(self, beta, x, y):
        N = len(x)
        xi, omega = beta[:N], beta[N:]
        g = np.dot(x, omega)
        g = y*g - 1 + xi
        return g
        
    def predict(self, x):
        return np.sign(np.dot(x, self.gamma))
    
    def error(self, x, y):
        return (len(x) - np.sum(y == self.predict(x)))/len(x)
    
    def model_selection(self, x_train, t_train, x_test, folds=5, num_basis=100):
        x_train, x_test = x_train.T, x_test.T
        t_train = t_train
        XC_dist, TC_dist, CC_dist, n, num_basis = dist(x_train, x_test, num_basis)
        # setup the cross validation
        cv_fold = np.arange(folds) # normal range behaves strange with == sign
        cv_split0 = np.floor(np.arange(n)*folds/n)
        cv_index = cv_split0[np.random.permutation(n)]
        # set the sigma list and lambda list
        sigma_list = np.array([0.01, 0.1, 1.])
        lda_list = np.array([0.01, 0.1, 1.])

        score_cv = np.zeros((len(sigma_list), len(lda_list)))

        for sigma_idx, sigma in enumerate(sigma_list):

            # pre-sum to speed up calculation
            h_cv = []
            t_cv = []
            for k in cv_fold:
                h_cv.append(np.exp(-XC_dist[:, cv_index==k]/(2*sigma**2)))
                t_cv.append(t_train[cv_index==k])

            for k in range(folds):
                #print(h0_cv[0])
                # calculate the h vectors for training and test
                count = 0
                for j in range(folds):
                    if j == k:
                        hte = h_cv[j].T
                        tte = t_cv[j]
                    else:
                        if count == 0:
                            htr = h_cv[j].T
                            ttr = t_cv[j]
                            count += 1
                        else:
                            htr = np.append(htr, h_cv[j].T, axis=0)
                            ttr = np.append(ttr, t_cv[j], axis=0)

                one = np.ones((len(htr),1))
                htr = np.concatenate([htr, one], axis=1)
                one = np.ones((len(hte),1))
                hte = np.concatenate([hte, one], axis=1)
                for lda_idx, lda in enumerate(lda_list):
                    self.C = lda

                    self.fit(htr, ttr.T)
                    score = self.error(hte, tte)

                    score_cv[sigma_idx, lda_idx] = score_cv[sigma_idx, lda_idx] + score

        (sigma_idx_chosen, lda_idx_chosen) = np.unravel_index(np.argmin(score_cv), score_cv.shape)
        sigma_chosen = sigma_list[sigma_idx_chosen]
        lda_chosen = lda_list[lda_idx_chosen]

        x_train = np.exp(-XC_dist/(2*sigma_chosen**2)).T
        x_test = np.exp(-TC_dist/(2*sigma_chosen**2)).T

        one = np.ones((len(x_train),1))
        x_train = np.concatenate([x_train, one], axis=1)
        one = np.ones((len(x_test),1))
        x_test = np.concatenate([x_test, one], axis=1)

        return x_train, x_test, lda_chosen

class AT():
    def __init__(self, C, epsilon):
        self.C = C
        self.epsilon = epsilon
        
    def fit(self, x, y):
        N, dim = x.shape
        beta0 = np.zeros(dim+N)
        
        fun = lambda beta: self.objective_function(beta, N)
        
        cons = ({'type': 'ineq', 'fun': lambda beta:  self.contraint_function(beta, x, y)},
                {'type': 'ineq', 'fun': lambda beta: beta[:N]})
        
        res = minimize(fun, beta0, method='SLSQP', constraints=cons)
        
        self.xi, self.gamma = res.x[:N], res.x[N:]
                
    def objective_function(self, beta, N):
        xi, omega = beta[:N], beta[N:]
        
        return np.sum(xi) + (self.C/2)*np.sqrt(np.mean(omega**2))
        
    def contraint_function(self, beta, x, y):
        N = len(x)
        xi, omega = beta[:N], beta[N:]
        g = np.dot(x, omega)
        g = y*g - self.epsilon*np.sum(np.abs(omega)) - 1 + xi
        return g

    def predict(self, x):
        return np.sign(np.dot(x, self.gamma))
    
    def error(self, x, y):
        return (len(x) - np.sum(y == self.predict(x)))/len(x)
    
    def model_selection(self, x_train, t_train, x_test, folds=5, num_basis=100):
        x_train, x_test = x_train.T, x_test.T
        t_train = t_train
        XC_dist, TC_dist, CC_dist, n, num_basis = dist(x_train, x_test, num_basis)
        # setup the cross validation
        cv_fold = np.arange(folds) # normal range behaves strange with == sign
        cv_split0 = np.floor(np.arange(n)*folds/n)
        cv_index = cv_split0[np.random.permutation(n)]
        # set the sigma list and lambda list
        sigma_list = np.array([0.01, 0.1, 1.])
        lda_list = np.array([0.01, 0.1, 1.])

        score_cv = np.zeros((len(sigma_list), len(lda_list)))

        for sigma_idx, sigma in enumerate(sigma_list):

            # pre-sum to speed up calculation
            h_cv = []
            t_cv = []
            for k in cv_fold:
                h_cv.append(np.exp(-XC_dist[:, cv_index==k]/(2*sigma**2)))
                t_cv.append(t_train[cv_index==k])

            for k in range(folds):
                #print(h0_cv[0])
                # calculate the h vectors for training and test
                count = 0
                for j in range(folds):
                    if j == k:
                        hte = h_cv[j].T
                        tte = t_cv[j]
                    else:
                        if count == 0:
                            htr = h_cv[j].T
                            ttr = t_cv[j]
                            count += 1
                        else:
                            htr = np.append(htr, h_cv[j].T, axis=0)
                            ttr = np.append(ttr, t_cv[j], axis=0)

                one = np.ones((len(htr),1))
                htr = np.concatenate([htr, one], axis=1)
                one = np.ones((len(hte),1))
                hte = np.concatenate([hte, one], axis=1)
                for lda_idx, lda in enumerate(lda_list):
                    self.C = lda

                    self.fit(htr, ttr.T)
                    score = self.error(hte, tte)

                    score_cv[sigma_idx, lda_idx] = score_cv[sigma_idx, lda_idx] + score

        (sigma_idx_chosen, lda_idx_chosen) = np.unravel_index(np.argmin(score_cv), score_cv.shape)
        sigma_chosen = sigma_list[sigma_idx_chosen]
        lda_chosen = lda_list[lda_idx_chosen]

        x_train = np.exp(-XC_dist/(2*sigma_chosen**2)).T
        x_test = np.exp(-TC_dist/(2*sigma_chosen**2)).T

        one = np.ones((len(x_train),1))
        x_train = np.concatenate([x_train, one], axis=1)
        one = np.ones((len(x_test),1))
        x_test = np.concatenate([x_test, one], axis=1)

        return x_train, x_test, lda_chosen

class MH():
    def __init__(self, C, D, cost, alpha, beta):
        self.C = C
        self.D = D
        self.cost = cost
        self.alpha = alpha
        self.beta = beta
        
    def fit(self, x, y):
        N, dim = x.shape
        params0 = np.zeros(2*dim+N)
        
        fun = lambda params: self.objective_function(params, x)
        
        cons = ({'type': 'ineq', 'fun': lambda params:  self.contraint_function0(params, x, y)},
                {'type': 'ineq', 'fun': lambda params:  self.contraint_function1(params, x, y)},
                {'type': 'ineq', 'fun': lambda params: params[:N]})
        
        res = minimize(fun, params0, method='SLSQP', constraints=cons)
        
        self.xi, self.gamma, self.theta = res.x[:N], res.x[N:-dim], res.x[-dim:]
                
    def objective_function(self, params, x):
        N, dim = x.shape
        xi, gamma, theta = params[:N], params[N:-dim], params[-dim:]
        
        obj = np.sum(xi) + (self.C/2)*np.sqrt(np.mean(gamma**2)) + (self.D/2)*np.sqrt(np.mean(theta**2))
                
        return obj
        
    def contraint_function0(self, params, x, y):
        N, dim = x.shape
        xi, gamma, theta = params[:N], params[N:-dim], params[-dim:]
        g = np.dot(x, theta)
        g = xi - self.cost*(1- self.beta*g)
        return g
    
    def contraint_function1(self, params, x, y):
        N, dim = x.shape
        xi, gamma, theta = params[:N], params[N:-dim], params[-dim:]
        
        g = np.array([np.dot(x[i], (theta/y[i] - gamma)) for i in range(len(y))])
        g = xi - 1- (self.alpha/2)*y*g
        
        return g
        
    def predict(self, x):
        return np.sign(np.dot(x, self.gamma))
    
    def reject(self, x):
        return np.sign(np.dot(x, self.theta))
    
    def error(self, x, y, x_reject, show_rate=False):
        N = len(x)
        wrong_answer = np.sum(y[self.reject(x_reject) == 1] != self.predict(x)[self.reject(x_reject) == 1])
        rejected = self.cost*np.sum(self.reject(x_reject) == -1)

        res1 = (wrong_answer + rejected)/N
        if show_rate:
            return res1, np.mean(self.reject(x_reject) == -1)
        else:
            return res1
    
    def model_selection(self, x_train, t_train, x_test, folds=5, num_basis=100, algorithm='Ridge', logit=False):
        x_train, x_test = x_train.T, x_test.T
        t_train = t_train
        XC_dist, TC_dist, CC_dist, n, num_basis = dist(x_train, x_test, num_basis)
        # setup the cross validation
        cv_fold = np.arange(folds) # normal range behaves strange with == sign
        cv_split0 = np.floor(np.arange(n)*folds/n)
        cv_index = cv_split0[np.random.permutation(n)]
        # set the sigma list and lambda list
        sigma_list = np.array([0.01, 0.1, 1])
        lda0_list = np.array([0.01, 0.1, 1])
        lda1_list = np.array([0.01, 0.1, 1])
        

        score_cv = np.zeros((len(sigma_list), len(lda0_list), len(lda1_list)))

        for sigma_idx, sigma in enumerate(sigma_list):

            # pre-sum to speed up calculation
            h_cv = []
            t_cv = []
            for k in cv_fold:
                h_cv.append(np.exp(-XC_dist[:, cv_index==k]/(2*sigma**2)))
                t_cv.append(t_train[cv_index==k])

            for k in range(folds):
                #print(h0_cv[0])
                # calculate the h vectors for training and test
                count = 0
                for j in range(folds):
                    if j == k:
                        hte = h_cv[j].T
                        tte = t_cv[j]
                    else:
                        if count == 0:
                            htr = h_cv[j].T
                            ttr = t_cv[j]
                            count += 1
                        else:
                            htr = np.append(htr, h_cv[j].T, axis=0)
                            ttr = np.append(ttr, t_cv[j], axis=0)

                one = np.ones((len(htr),1))
                htr = np.concatenate([htr, one], axis=1)
                one = np.ones((len(hte),1))
                hte = np.concatenate([hte, one], axis=1)
                
                for lda0_idx, lda0 in enumerate(lda0_list):
                     for lda1_idx, lda1 in enumerate(lda1_list):
                            
                        self.C = lda0
                        self.D = lda1
                        
                        self.fit(htr, ttr.T)
                        score = self.error(hte, tte, hte)
                        
                        score_cv[sigma_idx, lda0_idx, lda1_idx] = score_cv[sigma_idx, lda0_idx, lda1_idx] + score

        (sigma_idx_chosen, lda0_idx_chosen, lda1_idx_chosen) = np.unravel_index(np.argmin(score_cv), score_cv.shape)
        sigma_chosen = sigma_list[sigma_idx_chosen]
        lda0_chosen = lda0_list[lda0_idx_chosen]
        lda1_chosen = lda1_list[lda1_idx_chosen]

        x_train = np.exp(-XC_dist/(2*sigma_chosen**2)).T
        x_test = np.exp(-TC_dist/(2*sigma_chosen**2)).T

        one = np.ones((len(x_train),1))
        x_train = np.concatenate([x_train, one], axis=1)
        one = np.ones((len(x_test),1))
        x_test = np.concatenate([x_test, one], axis=1)

        return x_train, x_test, lda0_chosen, lda1_chosen

class ATRO():
    def __init__(self, C, D, cost, alpha, beta, epsilon):
        self.C = C
        self.D = D
        self.cost = cost
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        
    def fit(self, x, y):
        N, dim = x.shape
        params0 = np.zeros(2*dim+N)
        
        fun = lambda params: self.objective_function(params, x)
        
        cons = ({'type': 'ineq', 'fun': lambda params:  self.contraint_function0(params, x, y)},
                {'type': 'ineq', 'fun': lambda params:  self.contraint_function1(params, x, y)},
                {'type': 'ineq', 'fun': lambda params: params[:N]})
        
        res = minimize(fun, params0, method='SLSQP', constraints=cons)
        
        self.xi, self.gamma, self.theta = res.x[:N], res.x[N:-dim], res.x[-dim:]
                
    def objective_function(self, params, x):
        N, dim = x.shape
        xi, gamma, theta = params[:N], params[N:-dim], params[-dim:]
        
        obj = np.sum(xi) + (self.C/2)*np.sqrt(np.mean(gamma**2)) + (self.D/2)*np.sqrt(np.mean(theta**2))
                
        return obj
        
    def contraint_function0(self, params, x, y):
        N, dim = x.shape
        xi, gamma, theta = params[:N], params[N:-dim], params[-dim:]
        g = np.dot(x, theta)
        g = xi - self.cost*(1- self.beta*(g - self.epsilon*np.sum(np.abs(theta))))
        return g
    
    def contraint_function1(self, params, x, y):
        N, dim = x.shape
        xi, gamma, theta = params[:N], params[N:-dim], params[-dim:]
        g = np.array([np.dot(x[i], (theta/y[i] - gamma)) for i in range(len(y))])
        zeta = np.array([theta/y[i] - gamma for i in range(len(y))])
        g = xi - 1- (self.alpha/2)*(y*g + self.epsilon*np.sum(np.abs(zeta), axis=1))
        return g
        
    def predict(self, x):
        return np.sign(np.dot(x, self.gamma))
    
    def reject(self, x):
        return np.sign(np.dot(x, self.theta))
    
    def error(self, x, y, x_reject, show_rate=False):
        N = len(x)
        wrong_answer = np.sum(y[self.reject(x_reject) == 1] != self.predict(x)[self.reject(x_reject) == 1])
        rejected = self.cost*np.sum(self.reject(x_reject) == -1)

        res1 = (wrong_answer + rejected)/N

        if show_rate:
            return res1, np.mean(self.reject(x_reject) == -1)
        else:
            return res1
    
    def model_selection(self, x_train, t_train, x_test, folds=5, num_basis=100, algorithm='Ridge', logit=False):
        x_train, x_test = x_train.T, x_test.T
        t_train = t_train
        XC_dist, TC_dist, CC_dist, n, num_basis = dist(x_train, x_test, num_basis)
        # setup the cross validation
        cv_fold = np.arange(folds) # normal range behaves strange with == sign
        cv_split0 = np.floor(np.arange(n)*folds/n)
        cv_index = cv_split0[np.random.permutation(n)]
        # set the sigma list and lambda list
        sigma_list = np.array([0.01, 0.1, 1])
        lda0_list = np.array([0.01, 0.1, 1])
        lda1_list = np.array([0.01, 0.1, 1])
        

        score_cv = np.zeros((len(sigma_list), len(lda0_list), len(lda1_list)))

        for sigma_idx, sigma in enumerate(sigma_list):

            # pre-sum to speed up calculation
            h_cv = []
            t_cv = []
            for k in cv_fold:
                h_cv.append(np.exp(-XC_dist[:, cv_index==k]/(2*sigma**2)))
                t_cv.append(t_train[cv_index==k])

            for k in range(folds):
                #print(h0_cv[0])
                # calculate the h vectors for training and test
                count = 0
                for j in range(folds):
                    if j == k:
                        hte = h_cv[j].T
                        tte = t_cv[j]
                    else:
                        if count == 0:
                            htr = h_cv[j].T
                            ttr = t_cv[j]
                            count += 1
                        else:
                            htr = np.append(htr, h_cv[j].T, axis=0)
                            ttr = np.append(ttr, t_cv[j], axis=0)

                one = np.ones((len(htr),1))
                htr = np.concatenate([htr, one], axis=1)
                one = np.ones((len(hte),1))
                hte = np.concatenate([hte, one], axis=1)
                
                for lda0_idx, lda0 in enumerate(lda0_list):
                     for lda1_idx, lda1 in enumerate(lda1_list):
                            
                        self.C = lda0
                        self.D = lda1
                        
                        self.fit(htr, ttr.T)
                        score = self.error(hte, tte, hte)
                        
                        score_cv[sigma_idx, lda0_idx, lda1_idx] = score_cv[sigma_idx, lda0_idx, lda1_idx] + score

        (sigma_idx_chosen, lda0_idx_chosen, lda1_idx_chosen) = np.unravel_index(np.argmin(score_cv), score_cv.shape)
        sigma_chosen = sigma_list[sigma_idx_chosen]
        lda0_chosen = lda0_list[lda0_idx_chosen]
        lda1_chosen = lda1_list[lda1_idx_chosen]

        x_train = np.exp(-XC_dist/(2*sigma_chosen**2)).T
        x_test = np.exp(-TC_dist/(2*sigma_chosen**2)).T

        one = np.ones((len(x_train),1))
        x_train = np.concatenate([x_train, one], axis=1)
        one = np.ones((len(x_test),1))
        x_test = np.concatenate([x_test, one], axis=1)

        return x_train, x_test, lda0_chosen, lda1_chosen

def dist(x, T=None, num_basis=False):

    (d,n) = x.shape

    # check input argument

    if num_basis is False:
        num_basis = 100000

    idx = np.random.permutation(n)[0:num_basis]
    C = x[:, idx]

    # calculate the squared distances
    XC_dist = CalcDistanceSquared(x, C)
    TC_dist = CalcDistanceSquared(T, C)
    CC_dist = CalcDistanceSquared(C, C)

    return XC_dist, TC_dist, CC_dist, n, num_basis

def CalcDistanceSquared(X, C):
    '''
    Calculates the squared distance between X and C.
    XC_dist2 = CalcDistSquared(X, C)
    [XC_dist2]_{ij} = ||X[:, j] - C[:, i]||2
    :param X: dxn: First set of vectors
    :param C: d:nc Second set of vectors
    :return: XC_dist2: The squared distance nc x n
    '''

    Xsum = np.sum(X**2, axis=0).T
    Csum = np.sum(C**2, axis=0)
    XC_dist = Xsum[np.newaxis, :] + Csum[:, np.newaxis] - 2*np.dot(C.T, X)
    return XC_dist