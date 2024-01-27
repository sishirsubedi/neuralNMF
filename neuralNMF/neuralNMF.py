from .dutil import DataSet
from .util.logging import setlogger

from .model.net import nmfNET
from .dutil.netdl import load_data
from .dutil.read_write import save_model
from .model.loss import Recon_Loss_Func
import torch
import numpy as np 

import logging
logger = logging.getLogger(__name__)

class neuralNMF(object):
	def __init__(self,adata : DataSet,layers,device):
		self.adata = adata
		self.net = nmfNET(layers).to(device)
		self.device = device
  
	def train(self,epochs,lr):

		data_mem_size = self.adata.uns['data_mem_size']
		data = load_data(self.adata.X,self.adata.obs.values.flatten(),data_mem_size,self.device)
  
		self.adata.varm = {}
		self.adata.obsm = {}


		for xx,y in data:break

		thetalist,betalist = train_unsupervised(self.net,xx, epoch = epochs, lr = lr, weight_decay = 1, decay_epoch=1)

		self.adata.varm['betalist'] = betalist
		self.adata.obsm['thetalist'] = thetalist
		self.adata.uns['roworder'] = np.array(y)

   
	def save(self):
		save_model(self)


def train_unsupervised(net,X,epoch,lr, weight_decay, decay_epoch):
	
	loss_func = Recon_Loss_Func()

	thetalist = []

	betalist = None
				
	for i in range(epoch):
		net.zero_grad()
		betalist = net(X)
		loss = None
		loss = loss_func(net, X, betalist)
		loss.backward()

		if i % 10 == 0:
			logger.info('====> Epoch: {} Average loss: {:.4f}'.format(i, loss.data))
     
  
		for l in range(net.depth - 1):

			theta = net.nmflist[l].theta
			thetalist.append(theta)
			theta.data = theta.data.sub_(lr*theta.grad.data)
		
		if (i+1)%decay_epoch == 0:
			lr = lr*weight_decay

	return thetalist, betalist

