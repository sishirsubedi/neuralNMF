import asappy
import asapc 
import anndata as an

from sklearn.preprocessing import StandardScaler 

sample = 'sim'

wdir = 'simdata_asapp/'

data_size = 20000
number_batches = 1


asappy.create_asap_data(sample,working_dirpath=wdir)
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches,working_dirpath=wdir)

mtx=asap_object.adata.X.T

n_topics=13
seed=42
nmf_model = asapc.ASAPdcNMF(mtx,n_topics,seed)
nmfres = nmf_model.nmf()

scaler = StandardScaler()
beta_log_scaled = scaler.fit_transform(nmfres.beta_log)

total_cells = asap_object.adata.uns['shape'][0]

asap_object.adata.varm = {}
asap_object.adata.obsm = {}
asap_object.adata.uns['pseudobulk'] ={}
asap_object.adata.uns['pseudobulk']['pb_beta'] = nmfres.beta
asap_object.adata.uns['pseudobulk']['pb_theta'] = nmfres.theta


pred_model = asapc.ASAPaltNMFPredict(mtx,beta_log_scaled)
pred = pred_model.predict()
asap_object.adata.obsm['corr'] = pred.corr
asap_object.adata.obsm['theta'] = pred.theta

hgvs = asap_object.adata.var.genes
asap_adata = an.AnnData(shape=(len(asap_object.adata.obs.barcodes),len(hgvs)))
asap_adata.obs_names = [ x for x in asap_object.adata.obs.barcodes]
asap_adata.var_names = [ x for x in hgvs]

for key,val in asap_object.adata.uns.items():
    asap_adata.uns[key] = val

asap_adata.varm['beta'] = asap_object.adata.uns['pseudobulk']['pb_beta'] 
asap_adata.obsm['theta'] = asap_object.adata.obsm['theta']
asap_adata.obsm['corr'] = asap_object.adata.obsm['corr']

cluster_resolution = 0.5
asappy.leiden_cluster(asap_adata,resolution=cluster_resolution)
print(asap_adata.obs.cluster.value_counts())

asappy.plot_gene_loading(asap_adata,top_n=5,max_thresh=25)


asappy.run_umap(asap_adata,distance='euclidean',min_dist=0.1)
asappy.plot_umap(asap_adata,col='cluster',pt_size=2,ftype='png')

import pandas as pd
df = pd.DataFrame(asap_adata.obs)
df.reset_index(inplace=True)
df.to_csv(wdir+'results/asapp_label.csv',index=False)