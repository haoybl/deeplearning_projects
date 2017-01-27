# A Simple CNN for Microbe Image Classification

## Introduction
A simple CNN is constructed to perform classification of microbe images 
from three classes, Giardia, Defects and Crypto. 

Below are sample images from three classes:

Giardia:
![Giardia class](https://github.com/ntutangyun/deeplearning_projects/tree/master/proj_2_cnn_microbe_image_classificaton/sample_images/Giardia.jpg)

Defects:
![Defects class](https://github.com/ntutangyun/deeplearning_projects/tree/master/proj_2_cnn_microbe_image_classificaton/Defects.jpg)

Crypto:
![Crypto class](https://github.com/ntutangyun/deeplearning_projects/tree/master/proj_2_cnn_microbe_image_classificaton/Crypto.jpg)


## Image Processing & Data Augmentation
To save computation time and memory, a 400 X 400 pixels window are taken from center of each image. 
and downsampled to 100 X 100 pixel size. 

Since total number of images are less than 400. Thus insufficient to train the network. 
Images data are augmented through rotation and flipping+rotation:

Given each image, the image is rotated anticlockwise by 10 degree per step, all the way up to 350 degree, which gives 35 new images.
Then the image is up-down flipped, and rotated again, which produces another 35 new images. 

So in total 1 original image produces 72 images, including original and flipped image. 
So the orignal image dataset is augmented by 72 times, which gives around 30k images before training.

## Train, Test and Evaluation 
70% of images are randomly selected to be training data, and 15% for test and evaluation data each.

## Convolutional Neural Network

### Structure
The CNN is constructed in the following way:

First Convolutional Layer: 5x5 pixel patch, 1 color channel, 16 depth, ReLU activation function, 2x2 max pooling

Second Convolutional Layer: 5x5 pixel patch, 16 channel, 32 depth, ReLU, 2x2 max pooling

First Fully Connected Layer: 512 neurons, ReLU, dropout

Output layer: softmax, 3 neurons, ont-hot encoding. [class 1: 100, class 2:010, class 3:001]

### Hyperparameter
Learning rate: exponential Decay, initial learning rate = 0.05, decay_step=5000, decay_rate=0.8

Optimizer: AdagradOptimizer

Loss Function: reduced mean softmax cross entropy


## Description of Scripts

1. Prepare_img.m on [Matlab] 
   This file reads original images png files and stores image data in mat file.

2. img_data_augmentation.m [Matlab]
   This file performs data augmentation by methods mentioned above.This script makes use of function 
   defined in expand_by_rotation.m

3. prepare_img_dataset.ipynb [jupyter notebook, python]
   This script read augmented image data produced by img_data_augmentation.m, and associate labels to
   data for supervised training.

4. TF_CNN.py [python3.5 with Tensorflow]
   This Script uses tensorflow to construct the CNN for classification. 


## Result
The running result is shown below:

	step 0, Loss on Train Set  504.382
	step 0, training accuracy 46.875 %
	step 0, Validation accuracy 34.1 %
	step 500, Loss on Train Set  0.294405
	step 500, training accuracy 90.625 %
	step 500, Validation accuracy 91.5 %
	step 1000, Loss on Train Set  0.0197453
	step 1000, training accuracy 100 %
	step 1000, Validation accuracy 93.7 %
	step 1500, Loss on Train Set  0.146909
	step 1500, training accuracy 93.75 %
	step 1500, Validation accuracy 95.9 %
	step 2000, Loss on Train Set  0.144503
	step 2000, training accuracy 93.75 %
	step 2000, Validation accuracy 96.8 %
	step 2500, Loss on Train Set  0.105486
	step 2500, training accuracy 93.75 %
	step 2500, Validation accuracy 96.8 %
	step 3000, Loss on Train Set  0.027562
	step 3000, training accuracy 100 %
	step 3000, Validation accuracy 97.9 %

## Contact:

Email: ytang014@e.ntu.edu.sg

