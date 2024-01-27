import numpy as np
from scipy import stats
np.random.seed(42)

def sample_gamma(shape, rate, size):
	return np.random.gamma(shape, 1./rate, size=size)

def generate_H(N, K, alpha=1., eps=2.):
	H = sample_gamma(alpha, 1., size=(N,K))
	for k in range(K):
		size = H[k:k+1, int(k*N/K):int((k+1)*N/K)].shape
		H[k:k+1, int(k*N/K):int((k+1)*N/K)] =  sample_gamma(np.random.random(1)/10, 1./1000., size=size)
		H[k:k+1, int(k*N/K):int((k+1)*N/K)] = sample_gamma(alpha + eps, 1./eps, size=size)
	return H

def generate_W(P, K, noise_prop=0., beta=2., eps=4.):

	W = np.zeros((P, K))

    ## add noise
	P_0 = int((1. - noise_prop) * P)
	if noise_prop > 0.:
		size = W[(P-P_0):, :].shape
		W[(P-P_0):, :] = sample_gamma(0.7, 1., size=size)
	W[:P_0, :] = sample_gamma(beta, 1, size=(P_0, K))

	for k in range(K):
		size = W[int(k*P_0/K):int((k+1)*P_0/K), k:k+1].shape
		W[int(k*P_0/K):int((k+1)*P_0/K), k:k+1] = sample_gamma(np.random.random(1)/10, 1./1000., size=size)	
		W[int(k*P_0/K):int((k+1)*P_0/K), k:k+1] = sample_gamma(beta +eps, 1./eps, size=size)	
	
	return W


def generate_data(N,K,M,mode):    

    if mode=='block':    
        H = generate_H(N, K)
        W = generate_W(M, K)
        X = stats.poisson.rvs(H.dot(W.T))

    else:
        H = stats.gamma.rvs(0.5, scale=0.1, size=(N,K))
        W = stats.gamma.rvs(0.5, scale=0.1, size=(M,K))
        X = stats.poisson.rvs(H.dot(W.T))

    return H,W,X