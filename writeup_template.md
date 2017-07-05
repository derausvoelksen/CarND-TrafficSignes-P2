#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/number_of_samples_per_class.png "Number if Samples per Class"
[image2]: ./writeup/preprocessing_with_equalize_adapthist.PNG "Sample before and after preprocessing"
[image3]: ./writeup/chosen_samples.PNG "Chosen Samples"
[image4]: ./writeup/softmax.PNG "Model Certainty on Top 5 using Softmax"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed over the 43 classes.

![Number of Sampes per Class][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to use a grayscaled image as it reduces the amount of data to be processed by 2/3. In order to achive a good grayscaled image and enhance the image quality especially in light and dark parts of it, I used the equalizes_adapthist alogorithm contained in the scikit-image module.


Here is an example of a traffic sign image before and after grayscaling.

![Sample before and after preprocessing][image2]

As a last step, I normalized the image data by dividing by 255, subtracting 0.5 and divided again by 0.5 in order have values between [-1 and 1] so that SDG algorithm works best.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Greyscaled   							| 
| Convolution 5x5, ReLu     	| 5x5 stride with valid padding, outputs 28x28x20 	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x20 				|
| Convolution 5x5, RELU	    | 5x5 stride with valid padding, outputs 10x10x40	|
| Max pooling | 2x2 stride, outputs 5x5x40 |
| Flatten		| output 1000        									|
| Fully connected				| output 250, dropout 					|
| Fully Connected				| output 125, dropout			        |
| Fully Connected        		| output 84	, dropout					|
| Fully Connected        		| output 43								|
 
The Dropouts between the fully connected layers are for avoiding overfitting.

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam Optimizer. Batch Size was 128 in 10 Epochs. The learning rate is at first 0.001, in second run its 0.0001. Keep Probability is set ti 0.6.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.976
* test set accuracy of 0.976

If a well known architecture was chosen:
* I used the well known lenet architecture to start with, as it is a good architecture for symbol detection (handwriting). The lenet architecture achieved an accuracy of about 0.935. So I added additional dimensions of the layers as well as another fully connected layer.

 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Chosen Samples][image3]

I chose 4 photos of traffic signs with no disturbed background, and one actual photo of a traffic sign with some disturbances in terms of a realistic scenery. One of 4 images with no disturbung background (32, end of all speed and passing limits) has written text right in the center, so I wanted to see if I can fool the model and see, if the model considers this image of the sign to be a "End of no Passing" (And it did!). I expected the 4 chosen images showing almost nothing but the sign to be detected easily by the model (except the "end of no passing" as described before), and I expected the model to have trouble detecting the real-life image because of the disturbing background. However, my initial assumption seems to be wrong, as the detection accuracy does not seem to relate to the hypothesis.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority Road (12)	| Speed Limit 50 km/h 									| 
| Stop     			| Stop										|
| 60 km/h				| End of speed Limit (80km/h)											|
|End of all speed and passing limits	      		| end of no passing					 				|
| go straight or right			| Go straight or right     							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. I assume, that the given amount of training set was to low.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

![Model Certainty on Top 5 using Softmax][image4]
