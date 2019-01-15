
# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D 
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# variables
num_classes = 7 # NUmber of classes
batch_size = 128
epochs = 10

# Importing dataset
dataset = pd.read_csv("dataset\\fer2013.csv")
dataset.head()

X = dataset.iloc[:, 1].values #Depndent variable(Pixel Values of 48x48 image)
print(X[0])

y = dataset.iloc[:, 0].values #Independent variable
#Encoding multiple classification vairbales
y = keras.utils.to_categorical(y, num_classes)


# =============================================================================
# Data Preprocessing
# =============================================================================
'''
X[0] = X[0].split(" ")
pixels = np.array(X[0], 'float32')
'''
'''
pixels = np.zeros((d.shape[0], 48*48))
for i in range(pixels.shape[0]):
    t = X[i].split(' ')
    for j in range(pixels.shape[1]):
        pixels[i, j] = int(t[j])
'''
'''
num_of_instances = dataset.shape
z = []

X = X.astype(list)
for i in X:#range(num_of_instances):
    img = i.split(' ')
    pixels = np.array(img, 'float32')
    z.append(pixels)

z = np.array(z, 'float32')'''

# Values of pixels are in string format so needs to split the every single value of 48*48 pixels
split_list =[i.split() for i in X]

# Convert Spitted pixel values into numpy array of float type
pixels_value = np.array(split_list, 'float32')

# Normalize the values by dividing it with 255(max pixwl value)
pixels_value /= 255 #normalizing inputs between [0, 1]
""" or"""
# Per-channel normalization (subtract mean, divide by standard deviation)
pixels_value -= np.mean(pixels_value, axis=0)
pixels_value /= np.std(pixels_value, axis=0)

# Reshaping pixel values (48 = height, 48 = width, 1 = lightnes/darkness of pixel (0 = light, 1 = dark))
pixels_value = pixels_value.reshape(pixels_value.shape[0], 48, 48, 1)

# Changind data type of pixel values to float
pixels_value = pixels_value.astype('float32')

# =============================================================================
# Data Splitting
# =============================================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(pixels_value, y, test_size = 0.20)


# =============================================================================
# Model Building (CNN)
# =============================================================================

model = Sequential()

# First Convolution Layer
model.add(Conv2D(64, (5, 5), activation = 'relu', input_shape = (48, 48, 1)))
model.add(MaxPooling2D(pool_size = (5, 5), strides = (2, 2)))

# Second Convolution Layer
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(AveragePooling2D(pool_size = (3, 3), strides = (2, 2)))
'''
# Third Convolution Layer
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(AveragePooling2D(pool_size = (3, 3), strides = (2, 2)))
'''
# Flattening
model.add(Flatten())

# Fully Connected Neural Network (ANN)
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation ='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation = 'softmax'))

'''
gen = ImageDataGenerator()
train_generator = gen.flow(X_train, y_train, batch_size=256)
 
model.compile(loss='categorical_crossentropy' , optimizer=keras.optimizers.Adam(), 
              metrics=['accuracy'])
 
model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=5)


train_score = model.evaluate(X_train, y_train, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', 100*train_score[1])
 
test_score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', test_score[0])
print('Test accuracy:', 100*test_score[1])
'''

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs)

score = model.evaluate(X_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', 100*score[1])


# Saving Model
from keras.models import load_model
model.save('emo_detect.h5')# acc = 69, 56.78
# Loading save Model
#model = load_model('emo_detect.h5')

# Making the Confusion Matrix
y_pred = model.predict(X_test)
#y_pred = (y_pred > 0.5) 
y_pred = y_pred.round()

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))


# =============================================================================
# Testing on Custom Image
# =============================================================================


#function for drawing bar chart for emotion preditions
def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    
    plt.show()


img = image.load_img("smile.jpg", grayscale=True, target_size=(48, 48))
 
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
 
x /= 255
 
custom = model.predict(x)

emotion_analysis(custom[0])
 
x = np.array(x, 'float32')
x = x.reshape([48, 48]);
 
plt.gray()
plt.imshow(x)
plt.show()    