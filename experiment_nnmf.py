import neuralNMF
import logging 

sample = 'sim'
wdir = 'simdata/'

ds = 'simdata/data/sim.h5'
select_genes = neuralNMF.identify_hvgenes(ds,gene_var_z=1e-2)
print('high variable genes..',len(select_genes))

neuralNMF.create_dataset(sample,working_dirpath=wdir,select_genes=select_genes)

data_mem_size = 10000
layers = [25]
device = 'cpu'
epochs = 2000
model = neuralNMF.create_model(sample=sample,data_mem_size=data_mem_size,layers=layers,device=device,working_dirpath=wdir)

logging.info(model.net)

model.train(epochs=epochs,lr=1)
model.save()
