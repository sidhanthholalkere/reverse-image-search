from Model import Model
from Model import train
from load import load_file
from Model import accuracy


from noggin import create_plot

plotter, fig, ax = create_plot(metrics=["loss", "accuracy"])


learning_rate=0.01
model = Model(512, 50)
num_epochs = 25
batch_size = 32
margin = 0.1
path = r'data\resnet18_features.pkl'
triplets = all_triplets(path)


train(model, num_epochs, margin, triplets, learning_rate=learning_rate, batch_size=batch_size)
