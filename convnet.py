import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot = True)


'''This function is defined to initialize weight parameters 
   as variables in the TensorFlow graph with given shape'''
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

'''This function is defined to initialize bias parameters 
   as variables in the TensorFlow graph with given shape'''
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

'''Define a convolution that uses a stride of one and are zero 
   padded so that the output is the same size as the input.'''
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides = [1,1,1,1], padding='SAME')

'''Define a pooling operation as a plain old max pooling over 2x2 blocks'''
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')


def conv_net(iter_scale = 10000,restore = False, checkpoint_file = None, inputdata = mnist):
    with tf.Session() as sess:
        
        x = tf.placeholder(tf.float32, shape=[None,784])
        y_ = tf.placeholder(tf.float32, shape=[None,10])

        #first convolution layer and pooling layer
        # the convolution layer has 1 input channel and 32 output channels
        # Neurons in the convolution layer use ReLU non-linearity to get rid of the noisy inputs
        # the pooling layer apply pooling over 2x2 blocks and reduce the image size by half.
        W_conv1 = weight_variable([5,5,1,32])
        b_conv1 = bias_variable([32])
        x_image = tf.reshape(x,[-1,28,28,1])
        h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        
        #second convolution layer and pooling layer
        # the convolution layer has 32 input channels and 64 output channels
        # Neurons in the convolution layer use ReLU non-linearity to get rid of the noisy inputs
        # the pooling layer apply pooling over 2x2 blocks and reduce the image size by half.
        W_conv2 = weight_variable([5,5,32,64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        
        #first fully-connected layer
        #use ReLU non-linearity
        #use dropout method to prevent the neural network from overfitting
        W_fc1 = weight_variable([7*7*64,1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
        
        
        W_fc2 = weight_variable([1024,10])
        b_fc2 = bias_variable([10])
        
        #y_conv is the prediction given by our classifier
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        
        #use cross entropy function as our loss function
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
        saver = tf.train.Saver()
        
        #if the restore flag is True, restore variables from the pretrained model
        #and use the pre-trained parameters to predict labels for the input images
        if restore:
            print ("Loading variable from '%s'." % checkpoint_file)
            saver.restore(sess,checkpoint_file)
            print("test accuracy %g"%accuracy.eval(feed_dict={
                x: inputdata.test.images, y_: inputdata.test.labels, keep_prob: 1.0}))  
        #if the restore flag is False, we have to train the convolutional neural netowrk
        #to make it perform classification correctly.
        else:
            sess.run(tf.initialize_all_variables())
            for i in range(iter_scale):
                batch = inputdata.train.next_batch(50)
                if i%100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
                    print("step %d, training accuracy %g"%(i, train_accuracy))
                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            save_path = saver.save(sess,"my_model\model")
            print("test accuracy %g"%accuracy.eval(feed_dict={
                x: inputdata.test.images, y_: inputdata.test.labels, keep_prob: 1.0}))
        
        #after training, return the variables so that we can store them as a pickle file and reuse it later.
        return [sess.run(W_conv1),sess.run(b_conv1),sess.run(W_conv2),sess.run(b_conv2),sess.run(W_fc1),sess.run(b_fc1),sess.run(W_fc2),sess.run(b_fc2)]

if __name__ == "__main__":
    mnist = input_data.read_data_sets('MNIST_data',one_hot = True)
    parameters = conv_net(iter_scale = 10000,restore = False, checkpoint_file = 'my_model/model',inputdata = mnist)



