# Real-Time-Emotion-Detection

## Dataset

The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).  

Dataset contains two columns, "emotion" and "pixels". The "emotion" column contains a numeric code ranging from 0 to 6, inclusive, for the emotion that is present in the image. The "pixels" column contains a string surrounded in quotes for each image. The contents of this string a space-separated pixel values in row major order.

This dataset was prepared by Pierre-Luc Carrier and Aaron Courville, as part of an ongoing research project.

[Dataset link](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

## Preprocessing
**1)** Pixel values from dataset are in text format, so total 48x48 pixel values appear as single string.
These values are first splitted into single value making it array of 48x48 pixel values, then converted into float format.

**2) Normalization:**
There are differnet ways to normalize the pixel values,
1) Dividing every valu by 255(i.e. dividing by maximum pixel value)
2) Pre-channel Normalization : Where mean value is subtracted from every pixel value & then divided by standard deviation. 

## Model Building

CNN is used for trained the model.

Input: Pixel values of 48x48x1(h,w,ch)
Convolutional Layer: Total three Convolutional layers, one MaxPool and two AveragePool layers
Fully Connected Neural Network: Two layers with two dropout layers of 20%



