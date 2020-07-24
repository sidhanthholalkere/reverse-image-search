from Model import Model
from Model import train
from Model import accuracy



learning_rate=0.1
model = Model(512, 50)
num_epochs = 1
batch_size = 32
margin = 0.1
path = 'data/captions_train2014.json'



train(model, num_epochs, margin, path, learning_rate=learning_rate, batch_size=batch_size)
