import matplotlib.pylab as plt
import seaborn as sns 
import pandas as pd
import anndata as an
import neuralNMF as model

sample = 'sim'
wdir = ''

fpath = wdir+'results/'+sample
adata = an.read_h5ad(wdir+'results/'+sample+'.h5nnmf')

##plot theta
sns.clustermap(adata.uns['theta_l1'])
plt.savefig(fpath+'_theta1.png');plt.close()

sns.clustermap(adata.uns['theta_l2'])
plt.savefig(fpath+'_theta2.png');plt.close()

#plot beta
model.plot_gene_loading(adata = adata,col='beta_l1',top_n=10,max_thresh=50)

model.plot_gene_loading(adata = adata,col='beta_l2',top_n=10,max_thresh=50)


