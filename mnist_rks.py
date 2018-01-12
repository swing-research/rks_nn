import matplotlib
matplotlib.use('Agg') 
import numpy as np
import scipy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.layers as l
import matplotlib.pyplot as pt


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

img_size = 28
batch_size = 200 # m

layers = [32,64]
kernels = [4,4]
final_layer = 500

def random_cnn(x):
    inp=tf.reshape(x, (-1, img_size, img_size, 1))
    out = inp
    for i,h  in enumerate(layers):
        out = l.avg_pool2d(l.conv2d(out, h, 
            padding='valid',
            kernel_size = kernels[i],
            activation_fn = tf.nn.relu,
            trainable= False,
            reuse = tf.AUTO_REUSE,
            scope=('rand%d'%i)), 2)

    out = tf.reshape(out, (batch_size, -1))
    out = l.fully_connected(out, final_layer, trainable=False, activation_fn=tf.nn.sigmoid)

    return out


def train(x, y, loss_fn=tf.losses.hinge_loss):
    phi = random_cnn(x)

    # get size of phi
    size = tf.shape(x, 'phi_shape')[1]
    with tf.name_scope('loss'):
        alpha = tf.Variable(tf.random_uniform((final_layer,10)), 
            trainable=True, name='alpha')
        yhat = tf.matmul(phi, alpha)
        # yhat = phi

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(yhat,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        loss = loss_fn(y, yhat,weights =1.0/layers[-1])
        train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    return train_step, loss, alpha, accuracy


def plot_img(x, ax, title=''):
    ax.imshow(x.reshape(img_size, img_size))
    ax.set_title(title)


avg_mnist = np.mean(mnist.train.next_batch(10*batch_size)[0])
accuracies = []

with tf.Session() as sess:

    # specify input placeholders
    x = tf.placeholder(tf.float32, shape=[batch_size, img_size*img_size])
    y = tf.placeholder(tf.float32, shape=[batch_size, 10])

    op_train, op_get_loss, op_get_alpha, op_acc = train(x, y)

    sess.run(tf.global_variables_initializer())

    for t in range(200000):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs -= avg_mnist

        _, loss, alpha = sess.run([op_train, op_get_loss, op_get_alpha], feed_dict={x: batch_xs, y: batch_ys})
        if t%1000==0:
            print("Iteration %d, loss %f"%(t,loss))
            acc = sess.run(op_acc, feed_dict={x: batch_xs, y: batch_ys})
            print("accuracy: %f"%acc)
            print(np.linalg.norm(alpha, 2))
            print("\n")
            accuracies.append(acc*100)

    # plotting code
    pt.plot(accuracies)
    pt.ylabel('accuracy (%)')
    pt.xlabel('Epochs/1000')
    # pt.show()
    pt.savefig('acc.jpg',dpi=300)






 
