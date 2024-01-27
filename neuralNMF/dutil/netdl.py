from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from scipy import sparse
import numpy as np
import logging
logger = logging.getLogger(__name__)


class SparseDataset(Dataset):
	def __init__(self,mtx,label):
		self.mtx = mtx
		self.shape = mtx.shape
		self.label = label
  
	def __len__(self):
		return self.shape[0]

	def __getitem__(self, idx):
		return self.mtx[idx],self.label[idx]

def load_data(mtx,mtx_index,net_batch_size,device):

	logger.info('loading sparse data on...\n'+ device)

	device = torch.device(device)
	mtx = torch.tensor(mtx.astype(np.int32), dtype=torch.int32, device=device)

	return DataLoader(SparseDataset(mtx,mtx_index), batch_size=net_batch_size, shuffle=False)