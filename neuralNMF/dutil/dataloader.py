import pandas as  pd
import numpy as np
from scipy.sparse import csr_matrix
import h5py as hf
import tables
import glob
import os
import logging
logger = logging.getLogger(__name__)

class DataSet:
	def __init__(self,sample,sample_path):
		self.uns = {}
		self.uns['inpath'] = sample_path+'results/'+sample

	def get_datainfo(self):
		with tables.open_file(self.uns['inpath']+'.h5', 'r') as f:
			for group in f.walk_groups():
				if '/' not in  group._v_name:
					shape = getattr(group, 'shape').read()
					print(str(group)+'....'+str(shape))

	def get_dataset_names(self):
		datasets = []
		with tables.open_file(self.uns['inpath']+'.h5', 'r') as f:
			for group in f.walk_groups():
				if '/' not in  group._v_name:
					datasets.append(group._v_name)
		return datasets
	
	def _estimate_batch_mode(self,dataset_list,data_mem_size):

		self.uns['data_mem_size'] = data_mem_size
		self.uns['dataset_list'] = dataset_list
		self.uns['dataset_data_mem_size'] = {}

		f = hf.File(self.uns['inpath']+'.h5', 'r')

		if len(self.uns['dataset_list']) == 1:
			
			n_cells = f[self.uns['dataset_list'][0]]['shape'][()][0]

			if n_cells < self.uns['data_mem_size']:
				self.uns['run_full_data'] = True
				self.uns['dataset_data_mem_size'][self.uns['dataset_list'][0]] = n_cells

			else:
				self.uns['run_full_data'] = False
				self.uns['dataset_data_mem_size'][self.uns['dataset_list'][0]] = data_mem_size
		else:
			n_cells = 0 
			for ds in self.uns['dataset_list']:
				n_cells += f[ds]['shape'][()][0]

			if n_cells < self.uns['data_mem_size']:
				self.uns['run_full_data'] = True
				for ds in self.uns['dataset_list']:
					self.uns['dataset_data_mem_size'][ds] = f[ds]['shape'][()][0]

			else:
				self.uns['run_full_data'] = False
				for ds in self.uns['dataset_list']:
					self.uns['dataset_data_mem_size'][ds] = int(((f[ds]['shape'][()][0])/n_cells ) * data_mem_size)
		f.close()

	def initialize_data(self,dataset_list,data_mem_size):

		self._estimate_batch_mode(dataset_list, data_mem_size)

		f = hf.File(self.uns['inpath']+'.h5', 'r')

		total_datasets = len(dataset_list)
		genes = []
		shape = []
		barcodes = []
		mtx = None

		if total_datasets == 1 and self.uns['run_full_data'] :
			
			genes = [x.decode('utf-8') for x in f[self.uns['dataset_list'][0]]['genes'][()]]
			barcodes = [x.decode('utf-8') for x in f[self.uns['dataset_list'][0]]['barcodes'][()]]
			shape = [len(barcodes),len(genes)]
			f.close()

			mtx = self.load_full_data()

		elif total_datasets > 1 and self.uns['run_full_data'] :
			
			## get first dataset for gene list
			genes = [x.decode('utf-8') for x in f[self.uns['dataset_list'][0]]['genes'][()]]
			
			barcodes = []
			for ds in self.uns['dataset_list']:
				start_index = 0
				end_index = self.uns['dataset_data_mem_size'][ds]
				barcodes = barcodes + [x.decode('utf-8')+'@'+ds for x in f[ds]['barcodes'][()]][start_index:end_index]
			

			shape = [len(barcodes),len(genes)]
			f.close()

			mtx = self.load_full_data()

		elif total_datasets == 1 and not self.uns['run_full_data'] :
		
			genes = [x.decode('utf-8') for x in f[self.uns['dataset_list'][0]]['genes'][()]]
			len_barcodes = f[self.uns['dataset_list'][0]]['shape'][()][0]
			shape = [len_barcodes,len(genes)]
			
			barcodes = None
			mtx = None
			
			f.close()
		
		elif total_datasets > 1 and not self.uns['run_full_data'] :

			## get first dataset for gene list
			genes = [x.decode('utf-8') for x in f[self.uns['dataset_list'][0]]['genes'][()]]

			len_barcodes = 0
			for ds in self.uns['dataset_list']:
				len_barcodes += f[ds]['shape'][()][0]

			shape = [len_barcodes,len(genes)]
			
			barcodes = None
			mtx = None
			
			f.close()
		
		self.var = pd.DataFrame()
		self.var['genes']= genes

		self.uns['shape'] = shape

		if barcodes:
			self.obs = pd.DataFrame()
			self.obs['barcodes']= barcodes

		if isinstance(mtx, np.ndarray):
			self.X = mtx


	def load_datainfo_batch(self,batch_index,start_index, end_index):

		f = hf.File(self.uns['inpath']+'.h5', 'r')
		
		if len(self.uns['dataset_list']) == 1:
			ds = self.uns['dataset_list'][0]

			if self.uns['shape'][0] < end_index: end_index = self.uns['shape'][0] 

			barcodes = [x.decode('utf-8')+'@'+ds for x in f[ds]['barcodes'][()]][start_index:end_index]
			f.close()

			return barcodes

		else:

			barcodes = []
			for ds in self.uns['dataset_list']:

				end_index =  self.uns['dataset_data_mem_size'][ds] * batch_index
				start_index = 	end_index - self.uns['dataset_data_mem_size'][ds]			

				dataset_size = f[ds]['shape'][()][0]
				if dataset_size < end_index: end_index = dataset_size

				barcodes = barcodes + [x.decode('utf-8')+'@'+ds for x in f[ds]['barcodes'][()]][start_index:end_index]

			f.close()
			return barcodes
		
	def load_full_data(self):
		
		if len(self.uns['dataset_list']) == 1:
			
			start_index = 0
			end_index = self.uns['dataset_data_mem_size'][self.uns['dataset_list'][0]]

			with tables.open_file(self.uns['inpath']+'.h5', 'r') as f:
				for group in f.walk_groups():
					if self.uns['dataset_list'][0] == group._v_name:
						data = getattr(group, 'data').read()
						indices = getattr(group, 'indices').read()
						indptr = getattr(group, 'indptr').read()
						shape = getattr(group, 'shape').read()
						dataset_selected_gene_indices = getattr(group, 'dataset_selected_gene_indices').read()

						dat = []
						for ci in range(start_index,end_index,1):
							dat.append(np.asarray(
							csr_matrix((data[indptr[ci]:indptr[ci+1]], 
							indices[indptr[ci]:indptr[ci+1]], 
							np.array([0,len(indices[indptr[ci]:indptr[ci+1]])])), 
							shape=(1,shape[1])).todense()).flatten())
						
						dat = np.asarray(dat)
						dat = dat[:,dataset_selected_gene_indices]

						return dat
		else:
			mtx = []				
			with tables.open_file(self.uns['inpath']+'.h5', 'r') as f:

				# need to loop sample to match initialize data barcode order
				for ds_i,ds in enumerate(self.uns['dataset_list']):

					for group in f.walk_groups():
						if  ds == group._v_name:

							start_index = 0
							end_index = self.uns['dataset_data_mem_size'][ds]

							data = getattr(group, 'data').read()
							indices = getattr(group, 'indices').read()
							indptr = getattr(group, 'indptr').read()
							shape = getattr(group, 'shape').read()
							dataset_selected_gene_indices = getattr(group, 'dataset_selected_gene_indices').read()
							
							ds_dat = []
							for ci in range(start_index,end_index,1):
								ds_dat.append(np.asarray(
								csr_matrix((data[indptr[ci]:indptr[ci+1]], 
								indices[indptr[ci]:indptr[ci+1]], 
								np.array([0,len(indices[indptr[ci]:indptr[ci+1]])])), 
								shape=(1,shape[1])).todense()).flatten())
							
							ds_dat = np.asarray(ds_dat)
							ds_dat = ds_dat[:,dataset_selected_gene_indices]
							if ds_i == 0:
								mtx = ds_dat
							else:
								mtx = np.vstack((mtx,ds_dat))
			return mtx

	def load_data_batch(self,batch_index, start_index, end_index):
		
		if len(self.uns['dataset_list']) == 1:
			
			with tables.open_file(self.uns['inpath']+'.h5', 'r') as f:
				for group in f.walk_groups():
					if self.uns['dataset_list'][0] == group._v_name:
						data = getattr(group, 'data').read()
						indices = getattr(group, 'indices').read()
						indptr = getattr(group, 'indptr').read()
						shape = getattr(group, 'shape').read()
						dataset_selected_gene_indices = getattr(group, 'dataset_selected_gene_indices').read()

						dat = []
						for ci in range(start_index,end_index,1):
							dat.append(np.asarray(
							csr_matrix((data[indptr[ci]:indptr[ci+1]], 
							indices[indptr[ci]:indptr[ci+1]], 
							np.array([0,len(indices[indptr[ci]:indptr[ci+1]])])), 
							shape=(1,shape[1])).todense()).flatten())
						
						dat = np.asarray(dat)
						dat = dat[:,dataset_selected_gene_indices]

						return dat
		else:

			with tables.open_file(self.uns['inpath']+'.h5', 'r') as f:

				# need to loop sample to match initialize data barcode order
				for ds_i,ds in enumerate(self.uns['dataset_list']): 

					for group in f.walk_groups():
						
						if ds == group._v_name:

							logging.info('Loading data from...'+ds + ' batch : '+str(batch_index))

							## update index according to each sample 	
							end_index =  self.uns['dataset_data_mem_size'][ds] * batch_index
							start_index = 	end_index - self.uns['dataset_data_mem_size'][ds]			

							data = getattr(group, 'data').read()
							indices = getattr(group, 'indices').read()
							indptr = getattr(group, 'indptr').read()
							shape = getattr(group, 'shape').read()
							dataset_selected_gene_indices = getattr(group, 'dataset_selected_gene_indices').read()
							
							ds_dat = []

							if end_index >= shape[0] : end_index = shape[0] - 1

							for ci in range(start_index,end_index,1):
								ds_dat.append(np.asarray(
								csr_matrix((data[indptr[ci]:indptr[ci+1]], 
								indices[indptr[ci]:indptr[ci+1]], 
								np.array([0,len(indices[indptr[ci]:indptr[ci+1]])])), 
								shape=(1,shape[1])).todense()).flatten())
							
							ds_dat = np.asarray(ds_dat)
							ds_dat = ds_dat[:,dataset_selected_gene_indices]
							if ds_i == 0:
								logging.info(ds + ' batch : '+str(batch_index) + ' size :'+ str(ds_dat.shape))
								mtx = ds_dat
							else:
								mtx = np.vstack((mtx,ds_dat))
								logging.info(ds + ' batch : '+str(batch_index) + ' size :'+ str(ds_dat.shape))
			return mtx
			
	def construct_batch_df(self,size):
		mtx = self.load_data_batch(1,0,size)
		barcodes = self.load_datainfo_batch(1,0,size)
		df = pd.DataFrame(mtx)
		df.index = barcodes
		df.columns = self.var.genes.values
		return df

