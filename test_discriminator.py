import numpy as np
import tensorflow as tf
import utils
from keras.models import Model
from keras.layers import Input, Lambda, Concatenate
from functools import partial


sess = tf.Session()
epochs = 100
batch = 100
shuffle = False

class_one = utils.MNIST(binary = False)#, path = "../ib_autoencoders/datasets/")
class_one.shrink_supervised(per_label = 100, include_labels = [1], shuffle = shuffle)

class_two = utils.MNIST(binary = False)#, path = "../ib_autoencoders/datasets/")
class_two.shrink_supervised(per_label = 100, include_labels = [1], shuffle = shuffle)

n_samples = class_one.x_train.shape[0]

x1 = Input(shape = (class_one.dim,))
x2 = Input(shape = (class_two.dim,))
discrim = utils.Discriminator(x1, x2, input_dim = 784, layers = [200, 100], lr = 0.001)     
#d = partial(discrim)
pred = discrim() # outputs 2 prediction tensors
pred1, pred2 = pred
l1 = tf.zeros_like(pred1)
l2 = tf.ones_like(pred2)
loss = .5*utils.binary_crossentropy([l1, pred1]) + .5*utils.binary_crossentropy([l2, pred2])
trainer = tf.train.AdamOptimizer(0.001).minimize(loss)

with sess.as_default():
    tf.global_variables_initializer().run()
    for i in range(epochs):  # Outer training loop
        perm = np.random.permutation(n_samples) # random permutation of data for each epoch
        loss_avg = 0        

        for offset in range(0, (int(n_samples / batch) * batch), batch): 
            batch_one = class_one.x_train[perm[offset:(offset + batch)]]
            # second shuffle / perm?
            batch_two = class_two.x_train[perm[offset:(offset + batch)]]
            result = sess.run([trainer, loss],feed_dict = {x1: batch_one, x2: batch_two})
            loss_avg += np.mean(result[-1])


        print("Epoch ", i, " Loss: ", loss_avg/(n_samples/batch))