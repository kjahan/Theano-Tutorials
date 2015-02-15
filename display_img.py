import os, numpy as np
from PIL import Image

datasets_dir = '/media/datasets/'

def load_data():
    data_dir = os.path.join(datasets_dir,'mnist/')
    #load training data which has 60k images each 28*28 pixels
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    #skip the first 16 bytes and properly reshape the training data
    train_data = loaded[16:].reshape((60000,28*28)).astype(float)
    
    #load labels
    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    labels = loaded[8:].reshape((60000))
    return train_data, labels

def display_image(inx):
    train_data, label_data = load_data()
    data = ""
    for i in xrange( 28**2 ):
        data += chr(255 - int(train_data[inx][i])) + chr(255 - int(train_data[inx][i])) + chr(255 - int(train_data[inx][i]))
    im = Image.fromstring("RGB", (28,28), data)
    im.save("/Users/jahan/workspace/DeepLearning/hw_" + str(inx) + ".png", "PNG")
    print "label", label_data[inx]

display_image(2)
