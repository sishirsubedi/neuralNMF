import pandas as pd
import numpy as np
from plotnine import *
import seaborn as sns
import matplotlib.pylab as plt

from ..util.analysis import get_topic_top_genes,row_col_order
from .palette import get_colors


def plot_gene_loading(adata,col,top_n=3,max_thresh=100):
	df_beta = pd.DataFrame(adata.uns[col])
	df_beta.columns = adata.var.index.values
	df_beta = df_beta.loc[:, ~df_beta.columns.duplicated(keep='first')]
	df_top = get_topic_top_genes(df_beta.iloc[:,:],top_n)
	df_beta = df_beta.loc[:,df_top['Gene'].unique()]
	ro,co = row_col_order(df_beta)
	df_beta = df_beta.loc[ro,co]
	df_beta[df_beta>max_thresh] = max_thresh
	sns.clustermap(df_beta.T,cmap='viridis')
	plt.savefig(adata.uns['inpath']+'_'+col+'_th_'+str(max_thresh)+'.png');plt.close()
