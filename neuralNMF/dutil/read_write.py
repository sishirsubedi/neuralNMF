import pandas as  pd
import numpy as np
from scipy.sparse import csr_matrix
import h5py as hf
import tables
import json
import glob
import os
import logging
logger = logging.getLogger(__name__)


class CreateDatasetFromH5:

	def __init__(self,sample_path,sample):
		self.inpath = sample_path+'data/'
		self.outpath = sample_path+'results/'
		self.datasets = glob.glob(self.inpath+sample+'*.h5')


	def peek_datasets(self):
		for ds in self.datasets:
			f = hf.File(ds, 'r')
			print('Dataset : '+ os.path.basename(ds).replace('.h5','') 
	 		+' , cells : '+ str(f['matrix']['barcodes'].shape[0]) + 
			', genes : ' + str(f['matrix']['features']['id'].shape[0]))
			f.close()

	def merge_genes(self,select_genes = None):
		final_genes = []
		for ds_i, ds in enumerate(self.datasets):
			f = hf.File(ds, 'r')
			if ds_i ==0:
				final_genes = set([x.decode('utf-8') for x in list(f['matrix']['features']['id']) ])
			else:
				current_genes = set([x.decode('utf-8') for x in list(f['matrix']['features']['id']) ])
				final_genes = final_genes.intersection(current_genes)
			f.close()

		self.genes = list(final_genes)

		if isinstance(select_genes,list):
			self.genes = [ x for x in self.genes if x in select_genes]
		self.dataset_selected_gene_indices = {}
		self.dataset_selected_gene_names = {}

		for ds_i, ds in enumerate(self.datasets):
			f = hf.File(ds, 'r')
   
			current_genes = [x.decode('utf-8') for x in list(f['matrix']['features']['id'])]
			selected_gene_indices = [index for index, element in enumerate(current_genes) if element in self.genes]
			seleted_gene_names = [current_genes[x] for x in selected_gene_indices]
   
			self.dataset_selected_gene_indices[os.path.basename(ds).replace('.h5','')] = selected_gene_indices
			self.dataset_selected_gene_names[os.path.basename(ds).replace('.h5','')] = seleted_gene_names
		 
			f.close()
	
	def merge_data(self,fname):
	
		for ds_i,ds in enumerate(self.datasets):

			dataset = os.path.basename(ds).replace('.h5','')
			dataset_f = hf.File(ds, 'r')

			print('processing...'+dataset)
			
			if ds_i ==0:
				f = hf.File(self.outpath+fname+'.h5','w')
			else:
				f = hf.File(self.outpath+fname+'.h5','a')

			grp = f.create_group(dataset)

			grp.create_dataset('barcodes', data = dataset_f['matrix']['barcodes'] ,compression='gzip')

			current_genes = self.dataset_selected_gene_names[dataset]
			grp.create_dataset('genes',data=current_genes,compression='gzip')
			
			grp.create_dataset('indptr',data=dataset_f['matrix']['indptr'],compression='gzip')
			grp.create_dataset('indices',data=dataset_f['matrix']['indices'],compression='gzip')
			grp.create_dataset('data',data=dataset_f['matrix']['data'],compression='gzip')

			nr = len(dataset_f['matrix']['barcodes'])
			nc = dataset_f['matrix']['features']['id'].shape[0]
			
			data_shape = np.array([nr,nc])

			grp.create_dataset('shape',data=data_shape)
			
			grp.create_dataset('dataset_selected_gene_indices',data=self.dataset_selected_gene_indices[dataset],compression='gzip')

			f.close()

	def create_data(self,fname,select_genes=None):
			print('Generating common genes...')
			self.merge_genes(select_genes)
			print('Merging datasets...')
			print('processing...'+self.outpath+fname)
			self.merge_data(fname)
			print('completed.')

class CreateDatasetFromH5AD:
	def __init__(self,sample_path,sample):
		self.inpath = sample_path+'data/'
		self.outpath = sample_path+'results/'
		self.datasets = glob.glob(self.inpath+sample+'*.h5ad')

	def check_label(self):
		for ds in self.datasets:
			f = hf.File(ds, 'r')
			if '_index' not in f['obs'].keys():
				print(os.path.basename(ds))
				print(f['obs'].keys())
			f.close()

	def update_label(self):
		for ds in self.datasets:
			f = hf.File(ds, 'r+')
			if '_index' not in f['obs'].keys():
				print(os.path.basename(ds))
				if 'cell_id' in f['obs'].keys():
					f['obs']['_index'] = f['obs']['cell_id']
				elif 'index' in f['obs'].keys():
					f['obs']['_index'] = f['obs']['index']
			f.close()

	def peek_datasets(self):
		for ds in self.datasets:
			f = hf.File(ds, 'r')
			print('Dataset : '+ os.path.basename(ds).replace('.h5ad','') 
	 		+' , cells : '+ str(len(f['obs']['_index'])) + 
			', genes : ' + str(f['var']['feature_name']['categories'].shape[0]))
			f.close()

	def merge_genes(self,select_genes = None):
		final_genes = []
		for ds_i, ds in enumerate(self.datasets):
			f = hf.File(ds, 'r')
			if ds_i ==0:
				final_genes = set([x.decode('utf-8') for x in list(f['var']['feature_name']['categories']) ])
			else:
				current_genes = set([x.decode('utf-8') for x in list(f['var']['feature_name']['categories']) ])
				final_genes = final_genes.intersection(current_genes)
			f.close()

		self.genes = list(final_genes)
		if isinstance(select_genes,list):
			self.genes = [ x for x in self.genes if x in select_genes]
		self.dataset_selected_gene_indices = {}
		self.dataset_selected_gene_names = {}

		for ds_i, ds in enumerate(self.datasets):
			f = hf.File(ds, 'r')
   
			current_genes = [x.decode('utf-8') for x in list(f['matrix']['features']['id'])]
			selected_gene_indices = [index for index, element in enumerate(current_genes) if element in self.genes]
			seleted_gene_names = [current_genes[x] for x in selected_gene_indices]
   
			self.dataset_selected_gene_indices[os.path.basename(ds).replace('.h5','')] = selected_gene_indices
			self.dataset_selected_gene_names[os.path.basename(ds).replace('.h5','')] = seleted_gene_names
		 
			f.close()
	
	def merge_data(self,fname):
	
		for ds_i,ds in enumerate(self.datasets):

			dataset = os.path.basename(ds).replace('.h5ad','')
			dataset_f = hf.File(ds, 'r')

			print('processing...'+dataset)
			
			if ds_i ==0:
				f = hf.File(self.outpath+fname+'.h5','w')
			else:
				f = hf.File(self.outpath+fname+'.h5','a')

			grp = f.create_group(dataset)

			grp.create_dataset('barcodes', data = dataset_f['obs']['_index'] ,compression='gzip')
			current_genes = self.dataset_selected_gene_names[dataset]
			grp.create_dataset('genes',data=current_genes,compression='gzip')

			grp.create_dataset('indptr',data=dataset_f['X']['indptr'],compression='gzip')
			grp.create_dataset('indices',data=dataset_f['X']['indices'],compression='gzip')
			grp.create_dataset('data',data=dataset_f['X']['data'],compression='gzip')

		
			nr = len(dataset_f['obs']['_index']) 
			nc = dataset_f['matrix']['features']['id'].shape[0]
			
			data_shape = np.array([nr,nc])

			grp.create_dataset('shape',data=data_shape)
			
			grp.create_dataset('dataset_selected_gene_indices',data=self.dataset_selected_gene_indices[dataset],compression='gzip')

			f.close()

	def create_data(self,fname,select_genes = None):
			print('Generating common genes...')
			self.merge_genes(select_genes)
			print('Merging datasets...')
			self.merge_data(fname)
			print('completed.')

class CreateDatasetFromMTX:
	def __init__(self,inpath,sample_names):
		self.inpath = inpath
		self.samples = sample_names

	def merge_genes(self,filter_genes = None):
		final_genes = []
		for si,sample in enumerate(self.samples):
			print('processing...'+sample)
			df = pd.read_csv(self.inpath+sample+'/features.tsv.gz',sep='\t',header=None)
			if si == 0:
				final_genes = set([(x).replace('-','_') + '_' + (y).replace('-','_') for x,y in zip(df[0],df[1])])
			else:	
				current_genes = set([(x).replace('-','_') + '_' + (y).replace('-','_') for x,y in zip(df[0],df[1])])
				final_genes = final_genes.intersection(current_genes)
		if filter_genes != None:
			keep_genes = [x for x in list(final_genes) if x in filter_genes]
		self.genes = keep_genes

	
	def merge_data(self,fname):
	
		from scipy.io import mmread

		for si,sample in enumerate(self.samples):

			print('processing...'+sample)
			
			if si ==0:
				f = hf.File(self.outpath+fname+'.h5','w')
			else:
				f = hf.File(self.outpath+fname+'.h5','a')

			mm = mmread(self.inpath+sample+'/matrix.mtx.gz')
			mtx = mm.todense()
			
			df_rows = pd.read_csv(self.inpath+sample+'/features.tsv.gz',sep='\t',header=None)
			df_rows['gene'] = [(x).replace('-','_') + '_' + (y).replace('-','_') for x,y in zip(df_rows[0],df_rows[1])]

			df_cols = pd.read_csv(self.inpath+sample+'/barcodes.tsv.gz',sep='\t',header=None)

			df = pd.DataFrame(mtx)
			df.columns = df_cols[0].values
			df.index = df_rows['gene'].values
			df = df.T
			
			## filter preselected common genes
			print('pre-selection size..'+str(df.shape))
			df = df[self.genes].T
			print('post-selection size..'+str(df.shape))

			smat = csr_matrix(df.to_numpy())
			
			grp = f.create_group(sample)

			grp.create_dataset('barcodes',data=df_cols[0].values,compression='gzip')

			genes = [ str(x) for x in df.index.values]
			gene_names = [ str(x) for x in df.index.values]


			batch_label = [ sample for x in range(len(df.columns))]

			grp.create_dataset('batch_label',data=batch_label,compression='gzip')
			grp.create_dataset('genes',data=genes,compression='gzip')
			grp.create_dataset('gene_names',data=gene_names,compression='gzip')
			grp.create_dataset('indptr',data=smat.indptr,compression='gzip')
			grp.create_dataset('indices',data=smat.indices,compression='gzip')
			grp.create_dataset('data',data=smat.data,dtype=np.int32,compression='gzip')

			arr_shape = np.array([len(f[sample]['genes'][()]),len(f[sample]['barcodes'][()])])

			grp.create_dataset('shape',data=arr_shape)
			
			f.close()

			print('completed.')

def is_csr_or_csc(data, indptr, indices):
	num_rows = len(indptr) - 1
	num_cols = max(indices) + 1

	# Check for CSR format
	if len(data) == len(indices) and len(indptr) == num_rows + 1 and max(indices) < num_cols:
		return "CSR"

	# Check for CSC format
	if len(data) == len(indices) and len(indptr) == num_cols + 1 and max(indices) < num_rows:
		return "CSC"

	return "Not CSR or CSC"

def convertMTXtoH5AD(infile,outfile):

	dataset_f = hf.File(infile, 'r')

	print('processing...'+os.path.basename(infile))
	
	f = hf.File(outfile+'.h5ad','w')

	grp = f.create_group('X')
	grp.create_dataset('indptr',data=dataset_f['matrix']['indptr'],compression='gzip')
	grp.create_dataset('indices',data=dataset_f['matrix']['indices'],compression='gzip')
	grp.create_dataset('data',data=dataset_f['matrix']['data'],compression='gzip')

	grp = f.create_group('obs')
	barcodes = [x.decode('utf-8').replace('@','-') for x in list(dataset_f['matrix']['barcodes']) ]
	grp.create_dataset('_index', data = barcodes,compression='gzip')

	
	grp = f.create_group('var')
	g1 = grp.create_group('feature_name')
	g1.create_dataset('categories',data=dataset_f['matrix']['features']['id'],compression='gzip')
	
	f.close()
 
def convertH5toMTX(infile,outfile):
	import gzip
	from scipy.io import mmwrite

	f = hf.File(infile,'r')

	mtx_indptr = f['matrix']['indptr']
	mtx_indices = f['matrix']['indices']
	mtx_data = f['matrix']['data']
	barcodes = [x.decode('utf-8') for x in f['matrix']['barcodes']]
	features = [x.decode('utf-8') for x in f['matrix']['features']['id']]

	rows = csr_matrix((mtx_data,mtx_indices,mtx_indptr),shape=(len(barcodes),len(features)))
	mtx= rows.todense()
	mtx = mtx.T
	rows = csr_matrix(mtx)
	
	file_name = outfile+'.mtx.gz'
	with gzip.open(file_name, 'wb') as f: mmwrite(f, rows)
 
	pd.DataFrame(barcodes).to_csv(outfile+'.barcodes.csv.gz',compression='gzip')
	pd.DataFrame(features).to_csv(outfile+'.features.csv.gz',compression='gzip')
	

def save_model(model):
	import anndata as an
	adata = an.AnnData(shape=(len(model.adata.obs.barcodes),len(model.adata.var.genes)))
	adata.obs_names = [ x for x in model.adata.obs.barcodes]
	adata.var_names = [ x for x in model.adata.var.genes]
	
	for key,val in model.adata.uns.items():
		adata.uns[key] = val
	
	
	for bi,beta in enumerate(model.adata.varm['betalist']):
		adata.uns['beta'+'_l'+str(bi+1)] =beta.cpu().detach().numpy()
	   
	for ti,theta in enumerate(model.adata.obsm['thetalist']):
		adata.uns['theta'+'_l'+str(ti+1)] = theta.cpu().detach().numpy()
  
	adata.write_h5ad(adata.uns['inpath']+'.h5nnmf')


def write_h5(fname,row_names,col_names,smat):

	f = hf.File(fname+'.h5','w')

	grp = f.create_group('matrix')

	grp.create_dataset('barcodes', data = row_names ,compression='gzip')

	grp.create_dataset('indptr',data=smat.indptr,compression='gzip')
	grp.create_dataset('indices',data=smat.indices,compression='gzip')
	grp.create_dataset('data',data=smat.data,compression='gzip')

	data_shape = np.array([len(row_names),len(col_names)])
	grp.create_dataset('shape',data=data_shape)
	
	f['matrix'].create_group('features')
	f['matrix']['features'].create_dataset('id',data=col_names,compression='gzip')

	f.close()

def read_config(config_file):
	import yaml
	with open(config_file) as f:
		params = yaml.safe_load(f)
	return params

def data_fileformat(sample,sample_path):
	fpath = sample_path+'data/'+sample+'*'
	print('source ...'+fpath)
	datasets = glob.glob(fpath)
	ftypes = []
	for f in datasets:
		ftypes.append(f.split('.')[len(f.split('.'))-1])
	if len(np.unique(ftypes)) == 1:
		return ftypes[0]
	else:
		raise ValueError('Multiple file formats in the data directory.')

