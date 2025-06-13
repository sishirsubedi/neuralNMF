
############### generate data

import neuralNMF
import scipy
import torch
from neuralNMF.dutil.read_write import write_h5

H,W,X = neuralNMF.generate_data(N=1000,K=10,M=2000,mode='block')
smat = scipy.sparse.csr_matrix(X)
row_names = [ str(i) for i in range(X.shape[0])]
col_names = ['c'+str(i) for i in range(X.shape[1])]
write_h5('data/sim',row_names,col_names,smat)
