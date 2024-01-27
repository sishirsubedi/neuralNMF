from .neuralNMF import neuralNMF


from .model import net 
from .model.nmf import nmfl
from .model.loss import Recon_Loss_Func


from .dutil import DataSet, CreateDatasetFromH5, CreateDatasetFromH5AD, data_fileformat, save_model

from .plotting.plots import plot_gene_loading
from .util.sim import generate_data
from .util.logging import setlogger

from .util.hvgenes import identify_hvgenes


import logging
logger = logging.getLogger(__name__)

def create_dataset(sample,working_dirpath,select_genes=None):

	setlogger(sample=sample,sample_dir=working_dirpath)
	
	number_of_selected_genes = 0
	if isinstance(select_genes,list):
		number_of_selected_genes = len(select_genes)

	logging.info('Creating data... \n'+
	'sample :' + str(sample)+'\n'+
	'number_of_selected_genes :' + str(number_of_selected_genes)+'\n'
	)

	filetype = data_fileformat(sample,working_dirpath)
	
	## read source files and create dataset 
	if filetype == 'h5':
		ds = CreateDatasetFromH5(working_dirpath,sample) 
		print(ds.peek_datasets())
		ds.create_data(sample,select_genes) 
	elif filetype == 'h5ad':
		ds = CreateDatasetFromH5AD(working_dirpath,sample) 
		print(ds.peek_datasets())
		ds.create_data(sample,select_genes) 
	
	logging.info('Completed create data.')

def create_model(sample,data_mem_size,layers,device,working_dirpath):

	setlogger(sample,working_dirpath)


	## create anndata like object 
	adata = DataSet(sample,working_dirpath)
	dataset_list = adata.get_dataset_names()
	adata.initialize_data(dataset_list=dataset_list,data_mem_size=data_mem_size)
	
	layers = [adata.uns['shape'][0]] + layers


	logging.info('Creating  object... \n'+
		'data_mem_size :' + str(data_mem_size)+'\n'
		)
	logging.info('layers'+str(layers))
 
	return neuralNMF(adata,layers,device)


