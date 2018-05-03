from numpy import *
import pylab as pl

from scipy import stats
gauss = lambda x, mu, cov: stats.multivariate_normal.pdf(x, mean=mu, cov=cov, allow_singular=True)

import math
from plotGauss2D import *
from scipy.misc import logsumexp



def select_k(x, k):
    # randomly and uniformly select k element from x without replacement
    indexes = random.choice(len(x), k, replace=False)
    return array([x[i] for i in indexes])



class MOG:
    def __init__(self, K, pi=0, mu=0, cov=0):

        self.N, self.D = X.shape

        self.K = K

        # if parameter specified, use passed-in parameters
        self.mu = select_k(X, self.K) if not mu else mu
        self.cov = array([eye(self.D) * 0.1]*K) if not cov else cov
        self.pi = array([1/K]*K) if not pi else pi
    
        self.LL = self.log_likelihood()
        
    def print_param(self):
        print('pi')
        print(self.pi)
        print('mu')
        print(self.mu)
        print('cov')
        print(self.cov)
        print('r')
        print(self.r)
 
    def __str__(self):
        return "[pi=%.2f,mu=%s, cov=%s]"%(self.pi, self.mu.tolist(), self.cov.tolist())
    __repr__ = __str__
 
    colors = ('blue', 'yellow', 'black', 'red', 'cyan')
 
    def plot(self):
        assert len(self.colors) >= self.K
        fig = pl.figure()                   # make a new figure/window
        ax = fig.add_subplot(111, aspect='equal')
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        ax.set_xlim(min(x_min, y_min), max(x_max, y_max))
        ax.set_ylim(min(x_min, y_min), max(x_max, y_max))
        for (mu, cov, c) in zip(self.mu, self.cov, self.colors):
#             print(mu)
#             print(cov)
            e = plotGauss2D(mu, cov, color=c)
            ax.add_artist(e)
        pl.plot(X[:,0:1].T[0],X[:,1:2].T[0], 'gs') # plot data points
        pl.show()
        
    def e_step(self):
        # E step is basically just calculating r
        self.r = array([pi*gauss(X, mu, cov) for pi,mu,cov in zip(self.pi, self.mu, self.cov)])
        self.r /= sum(self.r, axis=0)
        
    def m_step(self):
        # M step updates pi, mu and cov
        # using the r value obtained from E step
        N_k = sum(self.r, axis=1)
        self.mu = array([sum(self.r[k][n]*x for n,x in enumerate(X))/N_k[k] for k in range(self.K)])
        
        # updating new cov depends on new mu, from bishop book
        # seperating cov from m_step method for possible child overwrite
        self.update_cov(N_k)
        self.pi = N_k / self.N
        
    def iterate(self):
        iterno = 0
        while True:
            
            self.e_step()
            self.m_step()
            ll = self.log_likelihood()
#             if not iterno%200:
#                 print(iterno)
#                 print(self.LL)
#                 print(ll)
#                 print(self.LL - ll)
            if self.LL - ll == 0: # converge until floating number tolerence
                return
            self.LL = ll
            if iterno > 1024:
                print('could not converge after 1024 iterations') # only used for cross-validation for 
                return
            iterno += 1
            
        
    def update_cov(self, N_k):
        cov = []
        for k in range(self.K):
            # not using oneliner because don't want to calculate d twice
            d = X-self.mu[k]
            cov.append(dot((self.r[k].reshape((self.N,1))*d).T, d)/N_k[k])
            
        self.cov = array(cov)
#         self.cov = array([sum([dot(self.r[k][n]*(x-self.mu[k]),(x-self.mu[k]).T) \
#                              for n,x in enumerate(self.X)], axis=0)/N_k[k] for k in range(self.K)])
        
    def log_likelihood(self):
        return sum(log(sum(pi*gauss(x, mu, cov) for pi,mu,cov in zip(self.pi,self.mu,self.cov))) for x in X)



def MOGs(X, Ks):
    # create h number of MOGs, so that we get different random initializations
    pool = [MOG(X, k) for k in K]
    
    thread_pool = ThreadPool(8)
    results = thread_pool.map(MOG.iterate, pool)
    
#     list(map(MOG.iterate, pool)) # iterate in paralle when cpu allows
    return pool[argmax(map(MOG.log_likelihood, pool))], pool



from multiprocessing.pool import Pool 



def tune_k(model, dataset_name):
    global X
    tune_id = dataset_name + '_K'
    X = loadtxt('data/'+dataset_name+'.txt')
    
    param_range = range(1,6) # hardcoding
    
    pool = [model(X, k) for k in param_range]
    
    for p in pool: p.iterate()
#     print(dataset_name, 'done')
    return [p.log_likelihood() for p in pool]
#     for p in pool: p.plot()
    
#     pl.plot(param_range, [p.log_likelihood() for p in pool])

#     pl.xlabel('No of Gauss')
#     pl.ylabel('Log Likelihood')
    
# #     pl.savefig(tune_id)
#     pl.show()



def random_within(i, j):
    # return random float within i,j
    return random.uniform(i, j, 1)[0]

def rand_x():
    # return a random x within range of X
    _, D = X.shape
    return [random_within(max(X.T[d]),min(X.T[d])) for d in range(D)]
            



def tune_mu(model, dataset_name):
    global X
    tune_id = dataset_name + '_K'
    X = loadtxt('data/'+dataset_name+'.txt')
    
    param_range = range(1,6) # hardcoding
    
    
    # first pool with normal (random x) models
    pool1 = [model(X, k) for k in param_range]
    
    
#     thread_pool = ThreadPool(8)
#     results = thread_pool.map(model.iterate, pool)
    for p in pool1: p.iterate()
    
    for p in pool1: p.plot()
    
    

    # second pool with random(x_min,x_max)
    mus = [[rand_x() for _ in range(k)] for k in param_range]
    pool2 = [model(X, k, mu=mu) for k,mu in zip(param_range,mus)]
    
#     thread_pool = ThreadPool(8)
#     results = thread_pool.map(model.iterate, pool)
    for p in pool2: p.iterate()
    for p in pool2: p.plot()
        
    pl.plot(param_range, [p.log_likelihood() for p in pool1])
    pl.plot(param_range, [p.log_likelihood() for p in pool2])    
    
    pl.xlabel('No of Gauss')
    pl.ylabel('Log Likelihood')
    pl.legend(['random x', 'random(x_min, x_max)'], loc='upper right')
#     pl.savefig(tune_id)
    pl.show()



# uncomment below code to see the graphs
# tune_k(MOG, 'data_2_large')
# tune_mu(MOG, 'data_2_large')



class MOG_diagonal(MOG):
    def update_cov(self, N_k):
        for k in range(self.K):
            self.cov[k] = zeros((self.D, self.D))
            for i in range(self.D):
                self.cov[k][i][i] = sum(self.r[k][n]*(X[n][i]-self.mu[k][i])**2 for n in range(self.N))/N_k[k]



# tune_k(MOG_diagonal, 'data_2_small')
# tune_k(MOG, 'data_2_small')



class K_means:
    def __init__(self, K):
        self.K = K
        self.mu = array([rand_x() for _ in range(self.K)])
        
    def assign(self):
        self.C = [[] for _ in range(self.K)]
        for x in X:
            self.C[argmin([linalg.norm(x-mu_k) for mu_k in self.mu])].append(x)
        if not all(self.C):
            self.__init__(self.K)
            self.assign()
        
    def update_mu(self):
        self.mu = array([sum(c,axis=0)/len(c) for c in self.C])
#         print(self.mu)
        
    def loss(self):
        return sum(min(linalg.norm(x-mu_k) for mu_k in self.mu) for x in X)
    
    def iterate(self):
        l = float('inf')
        while True:
            self.assign()
            self.update_mu()
            new_l = self.loss()
            if new_l == l:
                return self.mu
            l = new_l
            
class MOG_Kmeans(MOG):
    def __init__(self, X, K, pi=0, mu=0, cov=0):
        super().__init__(X, K, pi, mu, cov)
        k_means = K_means(K)
        self.mu = k_means.iterate()
        self.LL = self.log_likelihood()



import time



def measure_time(m, dataset):
    start = time.time()
    
    lls = tune_k(m, dataset)
    end = time.time()
    print(dataset, 'took', end-start, 'seconds')
    return end-start, lls



D = ['data_'+str(i) for i in (1,2,3)]
smalls = [d+'_small' for d in D]
larges = [d+'_large' for d in D]



# t1 = [measure_time(MOG, d)[0] for d in larges]



# t2 = [measure_time(MOG_Kmeans, d)[0] for d in larges]



# D = ('data_1_small', 'data_2_large', 'data_2_small', 'data_2_large', 'data_3_small', 'data_3_large')


# pl.xticks(range(len(D)), D)
# pl.plot(range(len(D)), t1)
# pl.plot(range(len(D)), t2)
# pl.xlabel('datasets')
# pl.ylabel('time taken to train over all K')
# pl.legend(['random init mu', 'K-means init mu'], loc='upper right')
# pl.savefig('runtime_k_means')
# pl.show()



def measure_ll(model, dataset, K):
    global X
    X = loadtxt('data/'+dataset+'.txt')
    m = model(X, K)
    m.iterate()
    return m.log_likelihood()



#### Model selection ####



def cross_validate(model, dataset, K, k):
    # return average log likelihood on k fold validation
    print('cross val {}, {}-fold, with K = {}'.format(dataset, k, K))
    global X
    D = loadtxt('data/'+dataset+'.txt')
    random.shuffle(D)
    D = split(D[:len(D)-len(D)%k], k)
    running_total = 0
    for n, val in enumerate(D):
        # make X everything except val
        X = None
        for d in D:
            
#             print(X)
            if d is val:
                continue
            if X is None:
                
                X = d
            else:
                concatenate((X, d),axis=0)
            
                
        # X is train and val is validation
        m = model(K)
        m.iterate()
        X = val
        ll = m.log_likelihood()
#         print(ll)
        running_total += ll # notice that now global X being val, likelihood is evaluated on the val
        
    return running_total/k
    



mog_perform = [[cross_validate(MOG, d, k, 5) for d in smalls] for k in range(1,6)]



mog_diag_perform = [[cross_validate(MOG_diagonal, d, k, 5) for d in smalls] for k in range(1,6)]



for r in array(mog_perform).T:
    pl.plot(range(1,6), r)
for r in array(mog_diag_perform).T:
    pl.plot(range(1,6), r)    

pl.xlabel('K')
pl.ylabel('validation log likelihood')
pl.legend(['data1_cov', 'data2_cov', 'data3_cov', 'data1_var', 'data2_var', 'data3_var'], loc='best')
pl.show()




cross_validate(MOG, 'data_2_small', 2, 4)



pl.plot(range(1,6), sum([r for r in array(mog_perform).T],axis=0))

pl.plot(range(1,6), sum([r for r in array(mog_diag_perform).T],axis=0))    

pl.xlabel('K')
pl.ylabel('sum of validation log likelihood')
pl.legend(['cov', 'var'], loc='best')
pl.show()





cross_val_perform = [[cross_validate(MOG, 'mystery_2', K, k) for K in range(1,4)] for k in range(3,12)]



cross_val_mog_diag_perform = [[cross_validate(MOG_diagonal, 'mystery_2', K, k) for K in range(1,4)] for k in range(3,12)]



pl.pcolor(cross_val_perform)
pl.xlabel('K')
pl.ylabel('k-folds')
pl.title('covariance matrix')
pl.colorbar()
pl.show()



pl.pcolor(cross_val_mog_diag_perform)
pl.xlabel('K')
pl.ylabel('k-folds')
pl.title('variance vector')
pl.colorbar()
pl.show()



pl.plot(range(2,6), sum([r for r in array(cross_val_perform).T],axis=0))

pl.plot(range(2,6), sum([r for r in array(cross_val_mog_diag_perform).T],axis=0))    

pl.xlabel('k (how many fold did cross validation use)')
pl.ylabel('sum of validation log likelihood')
pl.legend(['cov', 'var'], loc='best')
pl.show()

















































































