# Adversarial-Example-Generator
The Convolutional Networks works well in practice and across a wide range of visual recognition problems, but there are still many blind spots behind this model.
We constructed an adversarial example generator based on the MNIST dataset. The generator will introduce adversarial perturbations into the input images and make the CNN model misclassifies these images.

To reproduce the result, just run the 'adversarial_examply_generator.py' and load parameters from the pickle file'params.pkl'. The pickle file contains the pre-trained parameters of a convolutional neural network classifier. The file 'CONVNET.py' is provided so that users can train their own classifier to classify the hand written digits provided by the MNIST dataset.
