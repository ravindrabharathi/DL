# Visualize where the Convolutional Network is looking at using Grad-CAM for CIFAR10 dataset

When solving image classification problems , it would be helpful for us to know what information from the image is being used by the network to make its preditions. Grad-CAM is a way for us to visualize the pixels in the activation channels that contribute most actively to a certain prediction . This will help us to fine tune the model in such a way that it uses all/most of the information belonging to the object being classified while making a prediction as opposed to using only small parts of the object or on the fringes or background surrounding the object being classified. This will help the model learn more about the features of a particular class .

We will use a ResNet18 Model for the image classification task.

ResNet pre-trained models exist mostly for ImageNet datasets. Also the classic ResNet architecture is tuned more towards these larger datasets like ImageNet. The authors of the ResNet paper did some special experiments tuned towards CIFAR-10 dataset and the smallest of Networks using these recommendations turns out to be what the Keras Team calls a ResNet20 Model. 

So we have a couple of choices :

1. Use a Model as defined in section 4.2 or the ResNet paper - ResNet20 

2. Start with a Pretrained model with ImageNet weights , add/modify layers and train it again on the 32x32 sized CIFAR-10 images 

We will try both options for the classification task and then apply Grad-CAM on misclassified images from the resultant predictions

## Model 1 : ResNet20 as described in section 4.2 of [ResNet paper](https://arxiv.org/pdf/1512.03385.pdf)

Based on the recommendations in section 4.2 of the ResNet paper for CIFAR-10 dataset a Resnet20 Model has been defined in [https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py](https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py) . We will use ResNetv2 from this project for CIFAR-10 dataset .

We will make a few small changes:

Remove the last Dense layer

Add a Conv2D 1x1 to reduce the number of channels to 10 (reducer1)

The modified file is at https://raw.githubusercontent.com/srbharathee/DL/master/cifar10_resnet20.py

We will add the following image augmentations

1.Cutout/Random Eraser augmentation as defined in https://github.com/yu4u/cutout-random-erasing
2. random horizontal flip ,
3. random width_shift of .1
4. random height_shift of 0.1

After training the model for 100 epochs , we will do a prediction on the test dataset and get the first 50 misclassified images .
We will apply Grad-CAM visualization on these 50 misclassified images

The Notebook for this Grad-CAM visualization is at [https://github.com/srbharathee/DL/blob/master/CR20GC.ipynb](https://github.com/srbharathee/DL/blob/master/CR20GC.ipynb)

## Model 2 : Pre-trained ResNet18 from [https://github.com/qubvel/classification_models](https://github.com/qubvel/classification_models)
This pre-trained ResNet18 Model is based on ImageNet weights(CIFAR weights not available in this library). 

We make the following changes to the model :

1. Change the input shape to match that of CIFAR10 : 32,32,3

2. Add a 1x1 conv layer to squash the 512 channels of the pretrained model to 10 channels corresponding to 10 classes in CIFAR10

3. Add GlobalAveragePooling to convert these to 1D inputs suitable for the softmax prediction layer

4. Add softmax prediction

We will add random Cutout and Horizontal Flip Image Augmentation to train the model for 100 epochs 

We will then use the model with best validation accuracy for Image classification and get a set of 50 misclassified images . 

We will then apply Grad-CAM visualization on these 50 misclassified images and print the heatmaps 

This version of the model and Grad-CAM can be found at [https://github.com/srbharathee/DL/blob/master/CFRGC2.ipynb](https://github.com/srbharathee/DL/blob/master/CFRGC2.ipynb)



