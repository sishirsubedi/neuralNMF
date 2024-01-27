
'''
modified from ASAPP 

'''

import numpy as np
from scipy import special
from sklearn.preprocessing import StandardScaler

import logging
logger = logging.getLogger(__name__)

class altPMF():
    def __init__(self, n_components, max_iter=100, tol=1e-6,
                 smoothness=100, random_state=None,
                 **kwargs):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.smoothness = smoothness
        self.random_state = random_state
        self._parse_args(**kwargs)
        self._set_random()
        
    def _set_random(self):
        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)
        else:
            np.random.seed(42)
    
    def _compute_expectations(self, a, b):
        return (a/b, special.digamma(a) - np.log(b))        
    
    def _parse_args(self, **kwargs):
        self.b_a = float(kwargs.get('b_a', 0.1))

    def _init_beta(self, n_feats):
        self.beta_a = self.smoothness * np.random.gamma(self.smoothness, 1. / self.smoothness,size=(self.n_components, n_feats))
        self.beta_b = self.smoothness * np.random.gamma(self.smoothness, 1. / self.smoothness,size=(self.n_components, n_feats))
        self.Ebeta, self.Elogbeta = self._compute_expectations(self.beta_a, self.beta_b)

    def _init_theta(self, prev_theta,ep=1e-8):
                
        self.Etheta = prev_theta
        self.Elogtheta = np.log(prev_theta + ep)
    
    def _xexplog(self):
        return np.dot(np.exp(self.Elogtheta), np.exp(self.Elogbeta))

    def _update_xaux(self,X):
        self.xaux = X / self._xexplog()

    def _update_beta(self):
        self.beta_a = self.b_a +  np.exp(self.Elogbeta) * np.dot(np.exp(self.Elogtheta).T, self.xaux)
        
        self.beta_b = self.b_a + np.sum(self.Etheta, axis=0, keepdims=True).T
         
        self.Ebeta, self.Elogbeta = self._compute_expectations(self.beta_a, self.beta_b)
                
    def predict_beta(self,X, prev_theta):
        self.n_samples, self.n_feats = X.shape    
        self._init_theta(prev_theta)
        self._init_beta(self.n_feats)
        self._update_xaux(X)
        for _ in range(self.max_iter):
            self._update_xaux(X)
            self._update_beta()
        
        return {'ebeta':self.Ebeta,
                'elogbeta':self.Elogbeta
                }


    def _llk(self,X):
        return np.sum(X * np.log(self._xexplog()) - self.Etheta.dot(self.Ebeta))

'''
import neuralNMF
import scipy
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns 


H,W,X = neuralNMF.generate_data(N=100,K=10,M=200,mode='block')

model = altPMF(n_components=H.shape[1],max_iter=25)
res_beta = model.predict_beta(X,prev_theta=H)
W_hat = res_beta['ebeta']
x_hat = H @ W_hat

correlation_matrix = np.corrcoef(X, x_hat)
correlation_rows = correlation_matrix[:X.shape[0], x_hat.shape[0]:]
sns.heatmap(correlation_rows)
plt.savefig('pmf_corr.png');plt.close()

'''


