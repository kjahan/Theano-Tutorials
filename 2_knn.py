import numpy as np
from load import mnist
import random as rand

trX, teX, trY, teY = mnist(onehot=True)

#compute euclidean distamce between two images
def get_dist(img_1, img_2):
    return np.sqrt(sum(np.square(img_1 - img_2)))

#run knn
def knn(inx):
    distances = []
    img = teX[inx]  #test image
    for img_ in trX:
        distances.append(get_dist(img, img_))
    
    #predicted label
    #return trY[np.argmin(distances)]
    return np.where(trY[np.argmin(distances)] == 1.0)[0][0]

tp = 0
tot = 0
for cnt in xrange(100):
    tot += 1
    inx = rand.randint(1,9900)
    pred_label = knn(inx)
    true_label = np.where(teY[inx] == 1.0)[0][0]
    if pred_label == true_label:
        tp += 1
    print "inx=", inx, "predicted label=", pred_label, " true label=", true_label
print tp, tot
