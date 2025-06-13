import neuralNMF
import logging 

sample = 'sim'
wdir = ''
neuralNMF.create_dataset(sample,working_dirpath=wdir)

data_mem_size = 10000
layers = [20,10]
device = 'cpu'
epochs = 300
model = neuralNMF.create_model(sample=sample,data_mem_size=data_mem_size,layers=layers,device=device,working_dirpath=wdir)
logging.info(model.net)

model.train(epochs=epochs,lr=1)
model.save()
