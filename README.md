![png](images/architecture.png)
# Convolutional Neural Networks for Digit Recognition
This project is a walkthrough of my submission to Kaggle's [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) competition, which ranked in the top 12% of submissions. In this notebook we will build a powerful convolutional neural network (CNN) architecture to recognize and classify images of handwritten digits (0-9). For reference, my final classifier had a test **accuracy of 99.5%** upon submission to Kaggle. 

## Setup
We will start by importing essential libraries and loading our data. As I go through the notebook, I ensure that all imports are included in this first cell.


```
# Essential libraries
import pandas as pd
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
%matplotlib inline
import sys, time, datetime, cv2, os
from progressbar import ProgressBar

np.random.seed(0) # Set the random seed for reproducibility 
random_state = 0

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Tensorflow and Keras modules
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop, Adam

# Hyperopt
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials


sns.set(style = 'whitegrid', context='notebook', palette='deep')
mpl.rcParams['figure.figsize'] = (12,8)
```


```
train_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Projects/mnist/train.csv')
test_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Projects/mnist/test.csv')
```


```
# Redundant data copies to allow us to load in data faster if datasets have been altered
train = train_df.copy(deep=True)
test = test_df.copy(deep=True)
```


```
X_train = train.drop('label', axis=1)
y_train = train['label']
print('X_train shape: {}\ny_train shape: {}'.format(X_train.shape, y_train.shape))
```

    X_train shape: (42000, 784)
    y_train shape: (42000,)


## Data Cleaning and Preprocessing
In this next section, we will clean and preprocess our data to prepare it for our CNNs. This is a crucial step, as it will allow us to build an accurate classifier. 


```
g = sns.countplot(x = y_train)
g.set_title('Label Counts')
plt.show()
```


![png](images/mnist_7_0.png)


Fortunately we can see that the distribution of target variables is relatively even. If the target variable was heavily imbalanced, we would need to consider oversampling or undersampling techniques, but that is not the case. 


```
# Null and missing values
# Checking to see if any of the columns have missing values
print(X_train.isnull().sum(axis=0).any())
print(y_train.isnull().sum(axis=0).any())
print(test.isnull().sum(axis=0).any())
```

    False
    False
    False


Fortunately Kaggle has provided us with a nice and clean dataset that does not contain any null values. 

### Viewing the images
Each image in our dataset is represented by a row of 784 values. We can reshape these rows into 28x28 images and display them. Let's take a look at a few of the images in the dataset to get a sense for what we're working with.


```
plt.subplots(2, 5)

for i in range(10):
  plt.subplot(2, 5, i+1)
  instance = X_train.iloc[i]
  instance = instance.values.reshape(28,28)
  plt.imshow(instance, cmap='gray')

plt.show()
```


![png](images/mnist_12_0.png)


### Normalization
The pixels have values ranging from 0-255, but we can normalize these values to the range (0,1).


```
# Max value is 255
X_train.iloc[0].max()
```




    255




```
X_train = X_train/255.0
test = test/255.0
```

### Reshaping the data
As mentioned, we should reshape the data into square images, as they are currently represented by rows. 

The images are provided as 1D arrays of 784 values, which we will reshape to 28x28 arrays.


```
print('X_train shape: {}\nX_test shape: {}'.format(X_train.shape, test.shape))
```

    X_train shape: (42000, 784)
    X_test shape: (28000, 784)



```
# Reshaping the data
X_train = X_train.values.reshape(-1,28,28, 1)
test = test.values.reshape(-1, 28, 28, 1)
print('X_train shape: {}\nX_test shape: {}'.format(X_train.shape, test.shape))
```

    X_train shape: (42000, 28, 28, 1)
    X_test shape: (28000, 28, 28, 1)


### Training and validation data
To evaluate the performance of our CNNs when we eventually build the models, we will want to create a holdout/ validation set. Here, we will separate 20% of the values into a validation set. Additionally, we will stratify the y values so that the training and validation datasets have equal proportions of each of the 10 digits. 


```
y_train
```




    0        1
    1        0
    2        1
    3        4
    4        0
            ..
    41995    0
    41996    1
    41997    7
    41998    6
    41999    9
    Name: label, Length: 42000, dtype: int64




```
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train)
```


```
print('X_train shape: {}\nX_valid shape: {}\ny_train shape: {}\ny_valid shape: {}'.format(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape))
```

    X_train shape: (33600, 28, 28, 1)
    X_valid shape: (8400, 28, 28, 1)
    y_train shape: (33600,)
    y_valid shape: (8400,)


### Output Encoding


```
y_train.value_counts()
```




    1    3747
    7    3521
    3    3481
    9    3350
    2    3342
    6    3309
    0    3306
    4    3258
    8    3250
    5    3036
    Name: label, dtype: int64



We see that the images are labeled with values from 0-9, each label representing the digit in the image. If we use these values in a classifier, they will be recognized as ordered values, rather than categories. In order to train our CNN we will need to encode the output as categories, and we will use one-hot encoding to do so. This will create 10 columns of y values; each image will have a 1 in the column that represents the image and a 0 in every other column. Although `sklearn` offers an implementation, we will use `keras` to perform the one hot encoding.


```
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)
print('y_train shape: {}\ny_valid shape: {}'.format(y_train.shape, y_valid.shape))
```

    y_train shape: (33600, 10)
    y_valid shape: (8400, 10)



```
y_train[0]
```




    array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0.], dtype=float32)



As a result of our one-hot encoding, each output is represented by a vector of 10 values, with a value of 1 in the position of the output's label. 

## Base CNN Model
We will now train our first CNN models, establishing a paradigm with which we will train additional models and assess model performance. The base model will consist of 3 sets of convolutional and max pooling layers. The first convolutional layer will act as the input layer. Each convolutional layer will have double the filters of the previous convolutional layer. We will build our network this way to allow our CNN to initially detect local features and progressively begin to identify higher level features. 

Our convolutions serve to extract features and edges from our images. Following these layers, we will flatten the output and feed it into a fully connected layer. This layer will use the features to learn how to classify the images. Lastly, we will use a dense layer with 10 neurons, which will classify our images into each of the 10 categories, representing the digits.  


```
base_model = Sequential([
                         # Three sets of convolutional layers, followed by max pooling layers
                         # We will double the number of convolutional filters after each pooling layer
                         Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', input_shape=(28,28,1)),
                         MaxPool2D(padding='same'),

                         Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'),
                         MaxPool2D(padding='same'),
                         
                         Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'),
                         MaxPool2D(padding='same'),

                         # Fully connected layers to make predictions
                         Flatten(),
                         Dense(256, activation='relu'),
                         Dense(10, activation='softmax')
])
```


```
base_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

We will also incorporate early stopping into our model using a `keras` callback. This will allow us to set a large number of epochs and stop the training of the CNN once the validation accuracy stops improving. This will prevent overfitting of the model and return the model weights that resulted in the highest validation accuracy.


```
epochs = 30
batch_size = 32
early_stopping = EarlyStopping(min_delta=0.001, patience=5, restore_best_weights=True)
```


```
base_history = base_model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, callbacks=[early_stopping], validation_data=(X_valid, y_valid))
```

    Epoch 1/30
    1050/1050 [==============================] - 3s 3ms/step - loss: 0.1616 - accuracy: 0.9487 - val_loss: 0.0781 - val_accuracy: 0.9765
    Epoch 2/30
    1050/1050 [==============================] - 3s 3ms/step - loss: 0.0459 - accuracy: 0.9853 - val_loss: 0.0631 - val_accuracy: 0.9799
    Epoch 3/30
    1050/1050 [==============================] - 3s 3ms/step - loss: 0.0339 - accuracy: 0.9891 - val_loss: 0.0550 - val_accuracy: 0.9839
    Epoch 4/30
    1050/1050 [==============================] - 3s 3ms/step - loss: 0.0223 - accuracy: 0.9926 - val_loss: 0.0523 - val_accuracy: 0.9854
    Epoch 5/30
    1050/1050 [==============================] - 3s 3ms/step - loss: 0.0181 - accuracy: 0.9942 - val_loss: 0.0554 - val_accuracy: 0.9870
    Epoch 6/30
    1050/1050 [==============================] - 3s 3ms/step - loss: 0.0171 - accuracy: 0.9946 - val_loss: 0.0466 - val_accuracy: 0.9870
    Epoch 7/30
    1050/1050 [==============================] - 3s 3ms/step - loss: 0.0140 - accuracy: 0.9952 - val_loss: 0.0474 - val_accuracy: 0.9876
    Epoch 8/30
    1050/1050 [==============================] - 3s 3ms/step - loss: 0.0112 - accuracy: 0.9960 - val_loss: 0.0527 - val_accuracy: 0.9882
    Epoch 9/30
    1050/1050 [==============================] - 3s 3ms/step - loss: 0.0081 - accuracy: 0.9976 - val_loss: 0.0616 - val_accuracy: 0.9869
    Epoch 10/30
    1050/1050 [==============================] - 3s 3ms/step - loss: 0.0087 - accuracy: 0.9973 - val_loss: 0.0517 - val_accuracy: 0.9895
    Epoch 11/30
    1050/1050 [==============================] - 3s 3ms/step - loss: 0.0102 - accuracy: 0.9968 - val_loss: 0.0480 - val_accuracy: 0.9900



```
# Plotting the training and validation accuracy and loss
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.tight_layout(pad=5.0)

ax1.plot(base_history.history['accuracy'], label='Training Accuracy')
ax1.plot(base_history.history['val_accuracy'], label='Validation Accuracy')
legend = ax1.legend()
ax1.set_title('Training and Validation Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')

ax2.plot(base_history.history['loss'], label='Training Loss')
ax2.plot(base_history.history['val_loss'], label='Validation Loss')
legend = ax2.legend()
ax2.set_title('Training and Validation Loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')

plt.show()
```


![png](images/mnist_36_0.png)


Let's see how well our base model scored (in terms of validation accuracy).


```
base_score = (max(base_history.history['val_accuracy']))
print('Best Validation Accuracy: {:.4f}'.format(base_score))
```

    Best Validation Accuracy: 0.9900


Great, we managed to achieve a 99% accuracy with our base model. This is unsurprising as the task at hand is relatively simple. However, it also means that there is limited room for improvement. 

## Model Tuning
Now that we have created and evaluated our base model, we will attempt to strengthen the model's accuracy. To do so, we will edit the architecture as well as make use of data augmentation, which we will walk through below.  

### Data Augmentation

The first step to improving our model is to use **data augmentation**. Data augmentation is the process of artifically expanding the size of our training dataset, and in the case of image data, involves the creation of new samples that are slightly altered versions of training samples. We will use the `keras` `ImageDataGenerator`, which will not only allow us to augment our data, but also acts as a data generator that will feed our models with data sequentially rather than loading all of the data into RAM. This will give us a larger training dataset and help to improve model accuracy. 


```
datagen = ImageDataGenerator(
    rotation_range = 10, 
    zoom_range = 0.1, 
    width_shift_range = 0.1,
    height_shift_range = 0.1
)

datagen.fit(X_train)
```

The Keras ImageDataGenerator allows us to fit our training data, and we can use this data below as we fit our CNN models.

### Model Training
To improve our model, we will make two major changes.

First, rather than using sets of convolutional layers followed by pooling layers, we will use 2 sets of convolutional layers followed by a pooling in each stack. We will have 3 stacks of these layers as before. Conv-Pool and Conv-Conv-Pool architectures have been proven as some of the most powerful ways to build CNNs from scratch.

Second, we will add dropout layers to our model. These layers will randomly drop a portion of the data in each epoch. This prevents overfitting by helping to ensure that our models do not learn noise. 




```
model = Sequential([
                         # Three sets of convolutional layers, followed by max pooling layers
                         # This time, we are stacking two convolutional layers in each set
                         # We will double the number of convolutional filters after each pooling layer
                         Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(28,28,1)),
                         Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'),
                         MaxPool2D(strides=2),
                         Dropout(0.3),

                         Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),
                         Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),
                         MaxPool2D(strides=2, padding='same'),
                         Dropout(0.3),
                         
                         Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'),
                         Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'),
                         MaxPool2D(strides=2, padding='same'),
                         Dropout(0.3),

                         # Fully connected layers to make predictions
                         Flatten(),
                         Dense(256, activation='relu'),
                         Dropout(0.3),
                         Dense(10, activation='softmax')
])
```


```
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```


```
epochs = 30
batch_size = 128
early_stopping = EarlyStopping(min_delta=0.001, patience=10, restore_best_weights=True)
```


```
history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                              epochs=epochs,
                              callbacks=[early_stopping],
                              validation_data=(X_valid, y_valid))
```

    Epoch 1/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.5897 - accuracy: 0.8018 - val_loss: 0.1136 - val_accuracy: 0.9654
    Epoch 2/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.1341 - accuracy: 0.9577 - val_loss: 0.0833 - val_accuracy: 0.9776
    Epoch 3/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0909 - accuracy: 0.9724 - val_loss: 0.0446 - val_accuracy: 0.9869
    Epoch 4/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0702 - accuracy: 0.9784 - val_loss: 0.0477 - val_accuracy: 0.9867
    Epoch 5/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0657 - accuracy: 0.9801 - val_loss: 0.0557 - val_accuracy: 0.9846
    Epoch 6/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0554 - accuracy: 0.9836 - val_loss: 0.0259 - val_accuracy: 0.9933
    Epoch 7/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0506 - accuracy: 0.9840 - val_loss: 0.0304 - val_accuracy: 0.9913
    Epoch 8/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0492 - accuracy: 0.9850 - val_loss: 0.0360 - val_accuracy: 0.9910
    Epoch 9/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0460 - accuracy: 0.9861 - val_loss: 0.0269 - val_accuracy: 0.9924
    Epoch 10/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0425 - accuracy: 0.9874 - val_loss: 0.0344 - val_accuracy: 0.9907
    Epoch 11/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0443 - accuracy: 0.9870 - val_loss: 0.0408 - val_accuracy: 0.9917
    Epoch 12/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0407 - accuracy: 0.9877 - val_loss: 0.0238 - val_accuracy: 0.9939
    Epoch 13/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0416 - accuracy: 0.9881 - val_loss: 0.0241 - val_accuracy: 0.9932
    Epoch 14/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0417 - accuracy: 0.9878 - val_loss: 0.0356 - val_accuracy: 0.9923
    Epoch 15/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0423 - accuracy: 0.9884 - val_loss: 0.0371 - val_accuracy: 0.9895
    Epoch 16/30
    263/263 [==============================] - 9s 34ms/step - loss: 0.0407 - accuracy: 0.9882 - val_loss: 0.0309 - val_accuracy: 0.9925
    Epoch 17/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0382 - accuracy: 0.9888 - val_loss: 0.0241 - val_accuracy: 0.9946
    Epoch 18/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0434 - accuracy: 0.9889 - val_loss: 0.0297 - val_accuracy: 0.9926
    Epoch 19/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0400 - accuracy: 0.9887 - val_loss: 0.0243 - val_accuracy: 0.9936
    Epoch 20/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0413 - accuracy: 0.9882 - val_loss: 0.0318 - val_accuracy: 0.9917
    Epoch 21/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0390 - accuracy: 0.9890 - val_loss: 0.0325 - val_accuracy: 0.9925
    Epoch 22/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0421 - accuracy: 0.9886 - val_loss: 0.0272 - val_accuracy: 0.9932



```
# Plotting the training and validation accuracy and loss
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.tight_layout(pad=5.0)

ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
legend = ax1.legend()
ax1.set_title('Training and Validation Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylim(0.9, 1.0)

ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
legend = ax2.legend()
ax2.set_title('Training and Validation Loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylim(0.0, 0.1)

plt.show()
```


![png](images/mnist_50_0.png)



```
best_score = (max(history.history['val_accuracy']))
print('Best Validation Accuracy: {:.4f}'.format(best_score))
```

    Best Validation Accuracy: 0.9946


The updated CNN is doing quite well now that we have made some adjustments, including stacking two convolutional layers in each step. Let's see if we can take the model a bit further by replacing our early stopping callback with a ReduceLROnPlateau callback, which reduces the learning rate of the model's optimizer when the accuracy begins to plateau.

This callback will allow us to use all of the epochs we scheduled rather than stopping early, and will also potentially increase accuracy once the accuracy begins to plateau (as we saw with earlier executions, the accuracy plateaus before we are done with all 30 epochs).


```
lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=5, factor=0.1)
history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                              epochs=epochs,
                              callbacks=[lr_reduction],
                              validation_data=(X_valid, y_valid))
```

    Epoch 1/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0432 - accuracy: 0.9873 - val_loss: 0.0302 - val_accuracy: 0.9937
    Epoch 2/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0411 - accuracy: 0.9880 - val_loss: 0.0494 - val_accuracy: 0.9856
    Epoch 3/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0429 - accuracy: 0.9878 - val_loss: 0.0494 - val_accuracy: 0.9876
    Epoch 4/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0401 - accuracy: 0.9883 - val_loss: 0.0297 - val_accuracy: 0.9926
    Epoch 5/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0379 - accuracy: 0.9890 - val_loss: 0.0348 - val_accuracy: 0.9918
    Epoch 6/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0397 - accuracy: 0.9890 - val_loss: 0.0361 - val_accuracy: 0.9931
    Epoch 7/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0266 - accuracy: 0.9925 - val_loss: 0.0271 - val_accuracy: 0.9939
    Epoch 8/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0239 - accuracy: 0.9933 - val_loss: 0.0237 - val_accuracy: 0.9946
    Epoch 9/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0213 - accuracy: 0.9935 - val_loss: 0.0254 - val_accuracy: 0.9942
    Epoch 10/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0207 - accuracy: 0.9941 - val_loss: 0.0230 - val_accuracy: 0.9944
    Epoch 11/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0213 - accuracy: 0.9938 - val_loss: 0.0237 - val_accuracy: 0.9942
    Epoch 12/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0191 - accuracy: 0.9949 - val_loss: 0.0228 - val_accuracy: 0.9945
    Epoch 13/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0190 - accuracy: 0.9943 - val_loss: 0.0255 - val_accuracy: 0.9942
    Epoch 14/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0200 - accuracy: 0.9942 - val_loss: 0.0246 - val_accuracy: 0.9943
    Epoch 15/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0172 - accuracy: 0.9948 - val_loss: 0.0245 - val_accuracy: 0.9940
    Epoch 16/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0198 - accuracy: 0.9946 - val_loss: 0.0250 - val_accuracy: 0.9937
    Epoch 17/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0181 - accuracy: 0.9946 - val_loss: 0.0248 - val_accuracy: 0.9938
    Epoch 18/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0172 - accuracy: 0.9946 - val_loss: 0.0247 - val_accuracy: 0.9940
    Epoch 19/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0172 - accuracy: 0.9949 - val_loss: 0.0248 - val_accuracy: 0.9940
    Epoch 20/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0180 - accuracy: 0.9947 - val_loss: 0.0247 - val_accuracy: 0.9940
    Epoch 21/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0153 - accuracy: 0.9953 - val_loss: 0.0247 - val_accuracy: 0.9939
    Epoch 22/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0172 - accuracy: 0.9948 - val_loss: 0.0247 - val_accuracy: 0.9939
    Epoch 23/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0151 - accuracy: 0.9955 - val_loss: 0.0247 - val_accuracy: 0.9939
    Epoch 24/30
    263/263 [==============================] - 9s 34ms/step - loss: 0.0163 - accuracy: 0.9953 - val_loss: 0.0247 - val_accuracy: 0.9939
    Epoch 25/30
    263/263 [==============================] - 9s 34ms/step - loss: 0.0169 - accuracy: 0.9951 - val_loss: 0.0247 - val_accuracy: 0.9939
    Epoch 26/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0177 - accuracy: 0.9950 - val_loss: 0.0247 - val_accuracy: 0.9939
    Epoch 27/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0173 - accuracy: 0.9951 - val_loss: 0.0247 - val_accuracy: 0.9939
    Epoch 28/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0188 - accuracy: 0.9945 - val_loss: 0.0247 - val_accuracy: 0.9939
    Epoch 29/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0174 - accuracy: 0.9946 - val_loss: 0.0247 - val_accuracy: 0.9939
    Epoch 30/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.0182 - accuracy: 0.9947 - val_loss: 0.0247 - val_accuracy: 0.9939



```
# Plotting the training and validation accuracy and loss
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.tight_layout(pad=5.0)

ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
legend = ax1.legend()
ax1.set_title('Training and Validation Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
#ax1.set_ylim(0.9, 1.0)

ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
legend = ax2.legend()
ax2.set_title('Training and Validation Loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
#ax2.set_ylim(0.0, 0.1)

plt.show()
```


![png](images/mnist_54_0.png)



```
best_score = (max(history.history['val_accuracy']))
print('Best Validation Accuracy: {:.4f}'.format(best_score))
```

    Best Validation Accuracy: 0.9946


Thanks to the ReduceLROnPlateau callback, we were able to make a slight improvement on our model's validation accuracy!

## Bayesian Hyperparameter Optimization Using Hyperopt
We have made improvements to our CNN, but we can take it even further by optimizing our hyperparameters. In this project, we will Bayesian hyperparameter optimization with the `hyperopt` library. 

We will not fully explore the theory behind Bayesian optimization, but will provide a high-level overview of the concept. Unlike grid search and random search, which search through parameter spaces without learning anything about where the best values lie, Bayesian optimization builds a probability model of the objective function (in our case, validation accuracy) and uses this model to predict the optimal set of hyperparameters. Evaluating the true objective function is computationally expensive, and by using a probability model of the objective function as a proxy rather than using the objective function itself, Bayesian optimization is often able to converge far faster than traditional hyperparameter search methods. 

In the code below, we create the parameter space and use hyperopt to find the optimal set of parameters. After running this optimization, we must decode the hyperopt output, as the output presents parameter indices rather than the actual values.


```
hyperopt_space = {
    'dropout_1': hp.choice('dropout_1', [0.1, 0.2, 0.3, 0.4, 0.5]),
    'dropout_2': hp.choice('dropout_2', [0.1, 0.2, 0.3, 0.4, 0.5]),
    'dropout_3': hp.choice('dropout_3', [0.1, 0.2, 0.3, 0.4, 0.5]),
    'dropout_4': hp.choice('dropout_4', [0.1, 0.2, 0.3, 0.4, 0.5]),
    'optimizer': hp.choice('optimizer', ['adam', 'rmsprop']),
    'dense_neurons': hp.choice('dense_neurons', [128, 256, 512])
}


def hyperopt_cnn(pars):
    # print('Parameters: ', pars)

    # Instantiate Sequential model
    model = Sequential()

    # First convolutional stack
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     activation='relu', padding='same',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     activation='relu', padding='same'))
    model.add(MaxPool2D(strides=2))
    model.add(Dropout(pars['dropout_1']))

    # Second convolutional stack
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     activation='relu', padding='same'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     activation='relu', padding='same'))
    model.add(MaxPool2D(strides=2, padding='same'))
    model.add(Dropout(pars['dropout_2']))

    # Third convolutional stack
    model.add(Conv2D(filters=128, kernel_size=(3, 3),
                     activation='relu', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3),
                     activation='relu', padding='same'))
    model.add(MaxPool2D(strides=2, padding='same'))
    model.add(Dropout(pars['dropout_3']))

    # Classification and output stack
    model.add(Flatten())
    model.add(Dense(pars['dense_neurons'], activation='relu'))
    model.add(Dropout(pars['dropout_4']))
    model.add(Dense(10, activation='softmax'))

    # Compile
    model.compile(optimizer=pars['optimizer'], loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train
    lr_reduction = ReduceLROnPlateau(
        monitor='val_accuracy', patience=3)
    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                        epochs=30,
                        callbacks=[lr_reduction],
                        verbose=0,
                        validation_data=(X_valid, y_valid))

    # Record results of each epoch
    best_epoch = np.argmax(history.history['val_accuracy'])
    best_val_loss = np.min(history.history['val_loss'])
    best_val_acc = np.max(history.history['val_accuracy'])

    # Print results of each epoch
    print('Epoch {} - val acc: {} - val loss: {}'.format(
        best_epoch, best_val_acc, best_val_loss))
    sys.stdout.flush()

    # Return dictionary of results
    # Hyperopt will use the loss function provided by this dictionary
    # Using negative accuracy because hyperopt will try to minimize
    return {'loss': -best_val_acc, 'status': STATUS_OK}
    # return {'loss': -best_val_acc,
    #         'best_epoch': best_epoch,
    #         'eval_time': time.time() - start,
    #         'status': STATUS_OK, 'model': model, 'history': history}

# Perform the hyperparameter optimization using TPE algorithm
trials = Trials()
best = fmin(hyperopt_cnn, hyperopt_space, algo=tpe.suggest,
            max_evals=50, trials=trials)
print(best)
```

    Epoch 17 - val acc: 0.9950000047683716 - val loss: 0.02277952805161476
    Epoch 20 - val acc: 0.9953571557998657 - val loss: 0.02181493304669857
    Epoch 23 - val acc: 0.9955952167510986 - val loss: 0.019366076216101646
    Epoch 25 - val acc: 0.9955952167510986 - val loss: 0.0189706739038229
    Epoch 20 - val acc: 0.9952380657196045 - val loss: 0.02035650797188282
    Epoch 12 - val acc: 0.9947618842124939 - val loss: 0.022593185305595398
    Epoch 21 - val acc: 0.995119035243988 - val loss: 0.019744791090488434
    Epoch 15 - val acc: 0.9940476417541504 - val loss: 0.021519599482417107
    Epoch 18 - val acc: 0.9954761862754822 - val loss: 0.02332252822816372
    Epoch 28 - val acc: 0.9957143068313599 - val loss: 0.019047003239393234
    Epoch 15 - val acc: 0.9955952167510986 - val loss: 0.022858168929815292
    Epoch 14 - val acc: 0.995119035243988 - val loss: 0.021938325837254524
    Epoch 23 - val acc: 0.996071457862854 - val loss: 0.017964106053113937
    Epoch 23 - val acc: 0.9959523677825928 - val loss: 0.019798072054982185
    Epoch 19 - val acc: 0.9955952167510986 - val loss: 0.02014996111392975
    Epoch 11 - val acc: 0.9944047331809998 - val loss: 0.024412812665104866
    Epoch 26 - val acc: 0.995119035243988 - val loss: 0.023264912888407707
    Epoch 14 - val acc: 0.9950000047683716 - val loss: 0.020381862297654152
    Epoch 13 - val acc: 0.9948809742927551 - val loss: 0.02242767997086048
    Epoch 18 - val acc: 0.9954761862754822 - val loss: 0.021711796522140503
    Epoch 25 - val acc: 0.9958333373069763 - val loss: 0.017896637320518494
    Epoch 22 - val acc: 0.9953571557998657 - val loss: 0.019256414845585823
    Epoch 23 - val acc: 0.9959523677825928 - val loss: 0.018283728510141373
    Epoch 16 - val acc: 0.9950000047683716 - val loss: 0.019424648955464363
    Epoch 25 - val acc: 0.996071457862854 - val loss: 0.017589209601283073
    Epoch 19 - val acc: 0.9952380657196045 - val loss: 0.019867146387696266
    Epoch 26 - val acc: 0.9957143068313599 - val loss: 0.018495792523026466
    Epoch 12 - val acc: 0.9955952167510986 - val loss: 0.018080448731780052
    Epoch 18 - val acc: 0.9954761862754822 - val loss: 0.0201480183750391
    Epoch 24 - val acc: 0.9954761862754822 - val loss: 0.018689796328544617
    Epoch 23 - val acc: 0.9963095188140869 - val loss: 0.015669632703065872
    Epoch 11 - val acc: 0.9947618842124939 - val loss: 0.019807277247309685
    Epoch 28 - val acc: 0.9961904883384705 - val loss: 0.017744556069374084
    Epoch 18 - val acc: 0.9950000047683716 - val loss: 0.01845642179250717
    Epoch 16 - val acc: 0.995119035243988 - val loss: 0.02013280615210533
    Epoch 21 - val acc: 0.9954761862754822 - val loss: 0.018457969650626183
    Epoch 23 - val acc: 0.9948809742927551 - val loss: 0.020451964810490608
    Epoch 28 - val acc: 0.9953571557998657 - val loss: 0.018539616838097572
    Epoch 18 - val acc: 0.9957143068313599 - val loss: 0.01798103004693985
    Epoch 20 - val acc: 0.9954761862754822 - val loss: 0.017533576115965843
    Epoch 14 - val acc: 0.9948809742927551 - val loss: 0.019740359857678413
    Epoch 23 - val acc: 0.9958333373069763 - val loss: 0.019245896488428116
    Epoch 23 - val acc: 0.9952380657196045 - val loss: 0.01856902614235878
    Epoch 20 - val acc: 0.9947618842124939 - val loss: 0.019685300067067146
    Epoch 20 - val acc: 0.9955952167510986 - val loss: 0.02026566118001938
    Epoch 24 - val acc: 0.9953571557998657 - val loss: 0.019967548549175262
    Epoch 24 - val acc: 0.9955952167510986 - val loss: 0.01960582472383976
    Epoch 16 - val acc: 0.994523823261261 - val loss: 0.021026697009801865
    Epoch 25 - val acc: 0.9957143068313599 - val loss: 0.01960369013249874
    Epoch 16 - val acc: 0.9953571557998657 - val loss: 0.020491160452365875
    100%|██████████| 50/50 [2:49:08<00:00, 202.97s/it, best loss: -0.9963095188140869]
    {'dense_neurons': 2, 'dropout_1': 4, 'dropout_2': 3, 'dropout_3': 1, 'dropout_4': 0, 'optimizer': 0}



```
best
```




    {'dense_neurons': 2,
     'dropout_1': 4,
     'dropout_2': 3,
     'dropout_3': 1,
     'dropout_4': 0,
     'optimizer': 0}



Using the indices above, we will now train our final model, save the model, and plot the model architecture.


```
# Using the output, we will now train our final model
model = Sequential()

# First convolutional stack
model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 activation='relu', padding='same',
                 input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(MaxPool2D(strides=2))
model.add(Dropout(0.5))

# Second convolutional stack
model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(MaxPool2D(strides=2, padding='same'))
model.add(Dropout(0.4))

# Third convolutional stack
model.add(Conv2D(filters=128, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(Conv2D(filters=128, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(MaxPool2D(strides=2, padding='same'))
model.add(Dropout(0.2))

# Classification and output stack
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
lr_reduction = ReduceLROnPlateau(
    monitor='val_accuracy', patience=3)
history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                    epochs=30,
                    callbacks=[lr_reduction],
                    verbose=1,
                    validation_data=(X_valid, y_valid))
```

    Epoch 1/30
    263/263 [==============================] - 9s 33ms/step - loss: 0.5249 - accuracy: 0.8235 - val_loss: 0.0704 - val_accuracy: 0.9781
    Epoch 2/30
    263/263 [==============================] - 8s 32ms/step - loss: 0.1313 - accuracy: 0.9579 - val_loss: 0.0501 - val_accuracy: 0.9846
    Epoch 3/30
    111/263 [===========>..................] - ETA: 4s - loss: 0.1049 - accuracy: 0.9678


```
# Save the model
model.save('model.h5')
```


```
model.summary()
```


```
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='best_model.png', show_shapes=True, show_layer_names=True)
```

## Results
We have managed to achieve a validation accuracy of 99.5%! This is very strong performance, and as of this notebook's creation, landed me in the top 12% of the Digit Recognizer leaderboard.

Now that we have finalized our model, we will use it to make predictions and export the result.

## Exporting Results


```
y_pred = model.predict(test)
y_pred = np.argmax(y_pred, axis=1)
y_pred
```




    array([2, 0, 9, ..., 3, 9, 2])




```
output = pd.DataFrame(columns=['ImageId', 'Label'])
```


```
output['ImageId'] = range(1, 1+len(test_df))
output['Label'] = y_pred
```


```
output
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ImageId</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>27995</th>
      <td>27996</td>
      <td>9</td>
    </tr>
    <tr>
      <th>27996</th>
      <td>27997</td>
      <td>7</td>
    </tr>
    <tr>
      <th>27997</th>
      <td>27998</td>
      <td>3</td>
    </tr>
    <tr>
      <th>27998</th>
      <td>27999</td>
      <td>9</td>
    </tr>
    <tr>
      <th>27999</th>
      <td>28000</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>28000 rows × 2 columns</p>
</div>




```
output.to_csv('mnist_submissions.csv', index=False)
```


```

```
