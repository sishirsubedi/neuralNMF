
# # # #### neuralnmf synthetic data 

# import scipy
# import torch

# data = scipy.io.loadmat('data/synthetic_noise.mat')
# X = data['X']

# smat = scipy.sparse.csr_matrix(X)
# row_names = [ str(i) for i in range(X.shape[0])]
# col_names = ['c'+str(i) for i in range(X.shape[1])]

# from neuralNMF.dutil.read_write import write_h5

# write_h5('simdata/data/sim',row_names,col_names,smat)



# ############### simulation data from asapp

import neuralNMF

import scipy
import torch

H,W,X = neuralNMF.generate_data(N=1000,K=10,M=2000,mode='block')

smat = scipy.sparse.csr_matrix(X)
row_names = [ str(i) for i in range(X.shape[0])]
col_names = ['c'+str(i) for i in range(X.shape[1])]

from neuralNMF.dutil.read_write import write_h5

write_h5('simdata/data/sim',row_names,col_names,smat)
