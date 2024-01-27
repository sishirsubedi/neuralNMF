import anndata as an
import neuralNMF as model

sample = 'sim'
wdir = 'simdata/'

fpath = wdir+'results/'+sample
adata = an.read_h5ad(wdir+'results/'+sample+'.h5nnmf')

import matplotlib.pylab as plt
import seaborn as sns 


import pandas as pd

# x = adata.uns['theta_l1'] @ adata.uns['beta_l1']
# df = pd.DataFrame(x)
# df.index = adata.uns['roworder']
# df.sort_index(inplace=True)
# sns.clustermap(x)
# plt.savefig(fpath+'demo2_x.png');plt.close()

sns.clustermap(adata.uns['theta_l1'])
plt.savefig(fpath+'_theta1.png');plt.close()

# sns.clustermap(adata.uns['theta_l2'])
# plt.savefig(fpath+'_theta2.png');plt.close()

model.plot_gene_loading(adata = adata,col='beta_l1',top_n=10,max_thresh=50)
# model.plot_gene_loading(adata = adata,col='beta_l2',top_n=10,max_thresh=100)



import pandas as pd
df = pd.DataFrame(adata.uns['theta_l1'])


import umap
		
umap_coords = umap.UMAP(n_components=2,min_dist=0.1).fit_transform(df.values)

df_umap = pd.DataFrame(umap_coords)
df_umap.columns = ['umap1','umap2']

dfl = pd.read_csv('/home/BCCRC.CA/ssubedi/projects/experiments/neuralNMF/simdata_asapp/results/asapp_label.csv')
ct = ['-'.join(x.split('_')[1:]) for x in dfl['index'].values]
df_umap['celltype'] = pd.Categorical(ct)

from neuralNMF.plotting.palette import colors_20
from plotnine import *
col='celltype'
pt_size=5
legend_size=7

p = (ggplot(data=df_umap, mapping=aes(x='umap1', y='umap2', color=col)) +
    geom_point(size=pt_size) +
    scale_color_manual(values=colors_20)  +
    guides(color=guide_legend(override_aes={'size': legend_size})))

p = p + theme(
    plot_background=element_rect(fill='white'),
    panel_background = element_rect(fill='white'))


p.save(filename = fpath+'_umap.png', height=8, width=15, units ='in', dpi=600)