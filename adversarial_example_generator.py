
import _pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


def pixel_initialization(shape):
    initial = tf.zeros(shape = shape)
    return tf.Variable(initial)

'''Define a convolution that uses a stride of one and are zero 
   padded so that the output is the same size as the input.'''
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides = [1,1,1,1], padding='SAME')

'''Define a pooling operation as a plain old max pooling over 2x2 blocks'''
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')


''' Instead of tuninig parameters in the ConvNet,we adjust the 
    pixels of the input image to generate adversarial examples.
    In this graph, the variables are input pixels x.'''
def get_adversarial_perturbation(params):
    adversarial_perturbation = []
    with tf.Session() as sess:
        x = pixel_initialization([1,28,28,1])
        
        #manually label the desired output as 6, apply backpropagation to find the
        #rate of change of getting 6 with respect to each input pixel.
        y_ = tf.constant([[0,0,0,0,0,0,1,0,0,0]])
        
        #first convolution layer and pooling layer
        # the convolution layer has 1 input channel and 32 output channels
        # Neurons in the convolution layer use ReLU non-linearity to get rid of the noisy inputs
        # the pooling layer apply pooling over 2x2 blocks and reduce the image size by half.
        W_conv1 = tf.placeholder(tf.float32, shape = [5,5,1,32])
        b_conv1 = tf.placeholder(tf.float32, shape = [32])
        h_conv1 = tf.nn.relu(conv2d(x,W_conv1)+b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        
        #second convolution layer and pooling layer
        # the convolution layer has 32 input channels and 64 output channels
        # Neurons in the convolution layer use ReLU non-linearity to get rid of the noisy inputs
        # the pooling layer apply pooling over 2x2 blocks and reduce the image size by half.
        W_conv2 = tf.placeholder(tf.float32, shape = [5,5,32,64])
        b_conv2 = tf.placeholder(tf.float32, shape = [64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        #first fully-connected layer
        #use ReLU non-linearity
        #use dropout method to prevent the neural network from overfitting
        W_fc1 = tf.placeholder(tf.float32, shape = [7*7*64,1024])
        b_fc1 = tf.placeholder(tf.float32, shape = [1024])
        h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

        W_fc2 = tf.placeholder(tf.float32, shape = [1024,10])
        b_fc2 = tf.placeholder(tf.float32, shape = [10])
        
        #y_conv is the prediction given by our classifier
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        
        #use cross entropy function as our loss function
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess.run(tf.initialize_all_variables())
        
        #run this optimization 100 iterations to get an adversarial perturbation
        for i in range(100):
            train_accuracy = accuracy.eval(feed_dict={W_conv1:params[0], b_conv1: params[1], W_conv2:params[2], b_conv2:params[3], W_fc1:params[4], b_fc1:params[5], W_fc2:params[6], b_fc2:params[7], keep_prob: 1.0})
            train_step.run(feed_dict={W_conv1:params[0], b_conv1: params[1], W_conv2:params[2], b_conv2:params[3], W_fc1:params[4], b_fc1:params[5], W_fc2:params[6], b_fc2:params[7], keep_prob: 1.0})
        adversarial_perturbation = sess.run(x)
    
    #apply fast gradient sign method to generate adversarial perturbation
    adversarial_perturbation = np.sign(adversarial_perturbation)*0.5
    return adversarial_perturbation



def fool_the_ConvNet(params,mnist_2_images,mnist_2_labels,adversarial_perturbation):
    with tf.Session() as sess:
        '''This function is a set up classifier, it takes a set of input and predict the labels for these input.
        The difference between this classifier and the previous one is that there is no variable in this CNN 
        classifier and we don't need to trian anything in this model.'''
        
        #x and y should be set up as a placeholder so that we can feed input images and labels into x and y.
        x = tf.placeholder(tf.float32,shape = [None,784])
        y_ = tf.placeholder(tf.float32, shape = [None,10])
        #adversarial_per is a placeholder that catches the input adversarial perturbation.
        adversarial_per = tf.placeholder(tf.float32, shape = [1,28,28,1])
        
        #Everything is the same as in the previous model except that 
        #we have to reshape the input images and add perturbation to them and 
        #we don't need to train anything in this model.
        W_conv1 = tf.placeholder(tf.float32, shape = [5,5,1,32])
        b_conv1 = tf.placeholder(tf.float32, shape = [32])
        x_image = tf.reshape(x,[-1,28,28,1]) + adversarial_per
        h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = tf.placeholder(tf.float32, shape = [5,5,32,64])
        b_conv2 = tf.placeholder(tf.float32, shape = [64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = tf.placeholder(tf.float32, shape = [7*7*64,1024])
        b_fc1 = tf.placeholder(tf.float32, shape = [1024])
        h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

        W_fc2 = tf.placeholder(tf.float32, shape = [1024,10])
        b_fc2 = tf.placeholder(tf.float32, shape = [10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess.run(tf.initialize_all_variables())
        
        #print the accuracy of predictions on the adversarial examples
        print("test accuracy on adversarial examples: %g "%accuracy.eval(feed_dict={x:mnist_2_images, y_:mnist_2_labels, adversarial_per: adversarial_perturbation,W_conv1:params[0], b_conv1: params[1], W_conv2:params[2], b_conv2:params[3], W_fc1:params[4], b_fc1:params[5], W_fc2:params[6], b_fc2:params[7], keep_prob: 1.0}))    
        print ('predictions made by the classifier:', sess.run(y_conv,feed_dict={x:mnist_2_images, y_:mnist_2_labels, adversarial_per: adversarial_perturbation,W_conv1:params[0], b_conv1: params[1], W_conv2:params[2], b_conv2:params[3], W_fc1:params[4], b_fc1:params[5], W_fc2:params[6], b_fc2:params[7], keep_prob: 1.0}
                ))

if __name__ == "__main__":
    mnist = input_data.read_data_sets('MNIST_data',one_hot = True)
    #load images of digit 2 from MNIST dataset
    mnist_2_images = [mnist.train.images[13],mnist.train.images[16],mnist.train.images[65],mnist.train.images[76],mnist.train.images[94],mnist.train.images[95],mnist.train.images[103],mnist.train.images[104],mnist.train.images[110],mnist.train.images[113]]
    mnist_2_labels = [mnist.train.labels[13],mnist.train.labels[16],mnist.train.labels[65],mnist.train.labels[76],mnist.train.labels[94],mnist.train.labels[95],mnist.train.labels[103],mnist.train.labels[104],mnist.train.labels[110],mnist.train.labels[113]]    
    
    #load parameters from pre-trained convnet model
    parameters = []    
    with open('params.pkl','rb') as f:
        parameters = _pickle.load(f)
    
    #generate adversarial perturbation
    adversarial_perturbation = get_adversarial_perturbation(parameters)
    
    #visualization the perturbation and adversarial examples
    print ('Visualization of adversarial perturbation')
    plt.imshow(adversarial_perturbation.reshape((28,28)),cmap = 'gray')
    plt.show()
    for i in range(0,len(mnist_2_images)):
        plt.imshow(mnist_2_images[i].reshape((28,28))+adversarial_perturbation.reshape((28,28)),cmap = 'gray')
        plt.show()
   
    #feed the adversarial examples into the classifier 
    #and print the predictions of the classifier on these adversarial examples
    fool_the_ConvNet(parameters,mnist_2_images,mnist_2_labels,adversarial_perturbation)



