from Model import Model
from Model import train
from load import load_file
from Model import accuracy
from extract_triplets import all_triplets

from noggin import create_plot

#only for jupyter notebook
plotter, fig, ax = create_plot(metrics=["loss", "accuracy"])

<<<<<<< HEAD
#getting model, model params, data
learning_rate=0.1
=======

learning_rate=0.01
>>>>>>> 740348e9258a1955b88378f6f4821b9adfad6f67
model = Model(512, 50)
num_epochs = 25
batch_size = 32
margin = 0.1
path = r'data\resnet18_features.pkl'
triplets = all_triplets(path)

#training the model!!
train(model, num_epochs, margin, triplets, learning_rate=learning_rate, batch_size=batch_size)


