

# Traffic Sign Recognition

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./images/visualization.png "Visualization"
[image2]: ./images/visualization_dist.png "Distribution of classes"
[image3]: ./images/visualization_mc.png "Most common classes"
[image4]: ./images/visualization_lc.png "Most common classes"
[image5]: ./images/training_graph.png "Training graph"
[image6]: ./images/sample_tests.png "Test samples"
[image7]: ./images/sample_tests_results.png "Test samples results"
[image8]: ./images/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/mpdivecha/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here are some visualizations of the data set. 

First, I show some random samples of the images present in the dataset. As can be seen, the images lose some details due to the low resolution. 

![alt text][image1]

The following is the distribution of classes in the training set. As can be seen, the distribution is highly uneven. This can be alleviated by augmenting the dataset (although dataset augmentation has not been performed in this project.)

![alt text][image2]

To better visualize the distribution, here I plot the most and least frequent classes in the dataset.

![alt text][image3]

![alt text][image4]



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The model has been trained on 3 channel RGB color images, so there's no conversion to grayscale. The only pre-processing used is normalization. Each image is normalized to have zero mean and unit variance using the statistics computed across the whole dataset and all channels. A couple of pre-processing techniques were explored but they didn't improve the performance above and beyond normalization, and in some cases, degraded it. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model was inspired from VGGNet with a few minor tweaks. First the number of convolutional layers is greatly reduced as the size of the input image is smaller than what the original VGGNet expects. Secondly, I have added dropout to the first and second fully connected layers.

A tabular representation of the final model looks like:

|      Layer      |               Description                |
| :-------------: | :--------------------------------------: |
|      Input      |            32x32x3 RGB image             |
| Convolution 3x3 | 1x1 stride, same padding, outputs 32x32x64 |
|      RELU       |                                          |
| Convolution 3x3 | 1x1 stride, same padding, outputs 28x28x64 |
|      RELU       |                                          |
|   Max pooling   |      2x2 stride,  outputs 14x14x64       |
| Convolution 3x3 | 1x1 stride, same padding, outputs 12x12x128 |
|      RELU       |                                          |
| Convolution 3x3 | 1x1 stride, same padding, outputs 10x10x128 |
|      RELU       |                                          |
|   Max pooling   |       2x2 stride,  outputs 5x5x128       |
|     Flatten     |               Outputs 3200               |
| Fully Connected |               Outputs 4096               |
|      RELU       |                                          |
|     Dropout     |           Keep probability 0.5           |
| Fully Connected |               Outputs 4096               |
|      RELU       |                                          |
|     Dropout     |           Keep probability 0.5           |
| Fully Connected |                Outputs 43                |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The model was trained for 100 epochs with a batch-size of 128. I used the ADAM optimizer with a learning-rate of 0.001 throughout training. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of 0.998
* validation set accuracy of 0.982
* test set accuracy of 0.964

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  * The first model I tried was LeNet.  
* What were some problems with the initial architecture?
  * It failed to give the required accuracy but the training and validation accuracy graphs pointed to the fact that model was insufficient in the number of parameters. This led to the next architecture.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  * The second and final architecture chosen was inspired from VGGNet. Compared to the original VGGNet architecture, my architecture contains only four convolutional layer and 2 max-pooling layers. Initially, dropout was not used in the architecture, however, the training overfit without any form of regularization present. 
* Which parameters were tuned? How were they adjusted and why?
  * The network was trained from scratch for 100 epochs without any transfer learning. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  * The model is deep enough so that the effective receptive field of the network at the first convolutional layer is big enough to span the entire image.
  * Dropout is an important part of the training network as it serves as a regularizer by preventing the co-adaption of the units in the fully connected layers. 
  * Despite the dropout layer, there's evidence that the model might be over-parameterized. This can be alleviated by introducing more regularization. One technique is to use data augmentation. 

If a well known architecture was chosen:
* What architecture was chosen?
  * VGGNet was chosen for this project.
* Why did you believe it would be relevant to the traffic sign application? 
  * It was chosen because it is a great step-up from LeNet, can be prepared easily and will run efficiently on my machine. Also the dataset is not large enough to warrant a more complex model.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? 
  * The model gave a training set accuracy of 0.998, a validation set accuracy of 0.986 and a test set accuracy of 0.964. This shows that the model is working well for this dataset.


The following is the training graph for training and validation accuracies over 100 epochs of training.

![alt text][image5]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I have chosen to test my model on 10 traffic sign images. Here they are shown below: 

![alt text][image6] 

Most of the images initially had traffic signs "in the wild", i.e., the traffic signs were not centered possibly had some distortion. The images used here have been cropped that the traffic signs are centered and occupy a significant portion of the image.

In the above figure, the signs captioned "Not present" are signs that are not present in the dataset. Ideally, these should be classified as random predictions with low softmax scores. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image7]




The model was able to correctly guess 8 of the 10 traffic signs, which gives an accuracy of 80%.  The two images that were misclassified were not present in the dataset. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 37th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a priority road sign (probability of 1), and the image does contain a priority road sign. The top five soft max probabilities were

| Probability |      Prediction      |
| :---------: | :------------------: |
|    1.00     |    Priority road     |
|    0.00     | Speed limit (20km/h) |
|    0.00     | Speed limit (30km/h) |
|    0.00     | Speed limit (50km/h) |
|    0.00     | Speed limit (60km/h) |

These are the softmax probabilities for the second image. The prediction by the model is correct.

| Probability |      Prediction      |
| :---------: | :------------------: |
|    1.00     |      Ahead only      |
|    0.00     | Speed limit (20km/h) |
|    0.00     | Speed limit (30km/h) |
|    0.00     | Speed limit (50km/h) |
|    0.00     | Speed limit (60km/h) |

These are the softmax probabilities for the third image. The prediction by the model is correct.

| Probability |      Prediction      |
| :---------: | :------------------: |
|    1.00     |   Traffic signals    |
|    0.00     | Speed limit (20km/h) |
|    0.00     | Speed limit (30km/h) |
|    0.00     | Speed limit (50km/h) |
|    0.00     | Speed limit (60km/h) |

These are the softmax probabilities for the fourth image. The prediction by the model is correct.

| Probability |      Prediction      |
| :---------: | :------------------: |
|    1.00     | Speed limit (60km/h) |
|    0.00     | Speed limit (20km/h) |
|    0.00     | Speed limit (30km/h) |
|    0.00     | Speed limit (50km/h) |
|    0.00     | Speed limit (70km/h) |

These are the softmax probabilities for the fifth image. Although the prediction is wrong, the model is completely certain about its prediction. This is one of the drawbacks of deep learning, as in uncertainties aren't appropriately coded in the models.

| Probability |       Prediction        |
| :---------: | :---------------------: |
|   *1.00*    | *Wild animals crossing* |
|    0.00     |       Keep right        |
|    0.00     |  Speed limit (20km/h)   |
|    0.00     |  Speed limit (30km/h)   |
|    0.00     |  Speed limit (50km/h)   |

These are the softmax probabilities for the sixth image. The prediction by the model is correct.

| Probability |      Prediction      |
| :---------: | :------------------: |
|    1.00     | Speed limit (30km/h) |
|    0.00     | Speed limit (20km/h) |
|    0.00     | Speed limit (50km/h) |
|    0.00     | Speed limit (60km/h) |
|    0.00     | Speed limit (70km/h) |

These are the softmax probabilities for the seventh image. The prediction by the model is correct.

| Probability |          Prediction          |
| :---------: | :--------------------------: |
|    1.00     |        Slippery road         |
|    0.00     | Dangerous curve to the right |
|    0.00     |     Speed limit (20km/h)     |
|    0.00     |     Speed limit (30km/h)     |
|    0.00     |     Speed limit (50km/h)     |

These are the softmax probabilities for the eight image. Again the prediction is incorrect here, yet the model is fully certain about its prediction. 

| Probability |      Prediction      |
| :---------: | :------------------: |
|   *1.00*    |    *No vehicles*     |
|    0.00     | Go straight or right |
|    0.00     |   Turn left ahead    |
|    0.00     | Speed limit (20km/h) |
|    0.00     | Speed limit (30km/h) |

These are the softmax probabilities for the ninth image. The prediction by the model is correct.

| Probability |      Prediction       |
| :---------: | :-------------------: |
|    1.00     |      Bumpy road       |
|    0.00     |    Traffic signals    |
|    0.00     | Wild animals crossing |
|    0.00     | Speed limit (20km/h)  |
|    0.00     | Speed limit (30km/h)  |

These are the softmax probabilities for the tenth image. The prediction by the model is correct.

| Probability |          Prediction          |
| :---------: | :--------------------------: |
|    1.00     |      Children crossing       |
|    0.00     | Dangerous curve to the right |
|    0.00     |     Speed limit (20km/h)     |
|    0.00     |     Speed limit (30km/h)     |
|    0.00     |     Speed limit (50km/h)     |



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


