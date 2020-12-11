
## Setup


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

# Transfer Learning
from keras.applications import InceptionResNetV2
from keras.applications.resnet50 import ResNet50


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


```
g = sns.countplot(y_train)
g.set_title('Label Counts')
plt.show()
```

    /usr/local/lib/python3.6/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning



![png](mnist_files/mnist_6_1.png)



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


### Viewing the images


```
plt.subplots(2, 5)

for i in range(10):
  plt.subplot(2, 5, i+1)
  instance = X_train.iloc[i]
  instance = instance.values.reshape(28,28)
  plt.imshow(instance, cmap='gray')

plt.show()
```


![png](mnist_files/mnist_9_0.png)


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



We see that the images are labeled with values from 0-9, each label representing the digit in the image. In order to train our CNN we will need to encode the output as categories using one-hot encoding. We will do this using keras.


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

## CNN Models
We will now train our CNN models, starting with a base model and establishing a paradigm with which we will train additional models and assess model performance.

### Base CNN Model


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


```
epochs = 30
batch_size = 32
early_stopping = EarlyStopping(min_delta=0.001, patience=5, restore_best_weights=True)
```


```
base_history = base_model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, callbacks=[early_stopping], validation_data=(X_valid, y_valid))
```

    Epoch 1/30
    263/263 [==============================] - 1s 5ms/step - loss: 0.2547 - accuracy: 0.9198 - val_loss: 0.0909 - val_accuracy: 0.9730
    Epoch 2/30
    263/263 [==============================] - 1s 5ms/step - loss: 0.0655 - accuracy: 0.9787 - val_loss: 0.0582 - val_accuracy: 0.9826
    Epoch 3/30
    263/263 [==============================] - 1s 5ms/step - loss: 0.0404 - accuracy: 0.9876 - val_loss: 0.0573 - val_accuracy: 0.9829
    Epoch 4/30
    263/263 [==============================] - 1s 5ms/step - loss: 0.0304 - accuracy: 0.9903 - val_loss: 0.0597 - val_accuracy: 0.9812
    Epoch 5/30
    263/263 [==============================] - 1s 5ms/step - loss: 0.0229 - accuracy: 0.9929 - val_loss: 0.0465 - val_accuracy: 0.9870
    Epoch 6/30
    263/263 [==============================] - 1s 5ms/step - loss: 0.0173 - accuracy: 0.9944 - val_loss: 0.0583 - val_accuracy: 0.9827
    Epoch 7/30
    263/263 [==============================] - 1s 5ms/step - loss: 0.0146 - accuracy: 0.9949 - val_loss: 0.0685 - val_accuracy: 0.9811
    Epoch 8/30
    263/263 [==============================] - 1s 5ms/step - loss: 0.0116 - accuracy: 0.9962 - val_loss: 0.0494 - val_accuracy: 0.9879
    Epoch 9/30
    263/263 [==============================] - 1s 5ms/step - loss: 0.0104 - accuracy: 0.9963 - val_loss: 0.0440 - val_accuracy: 0.9890
    Epoch 10/30
    263/263 [==============================] - 1s 5ms/step - loss: 0.0088 - accuracy: 0.9971 - val_loss: 0.0519 - val_accuracy: 0.9867
    Epoch 11/30
    263/263 [==============================] - 1s 5ms/step - loss: 0.0091 - accuracy: 0.9972 - val_loss: 0.0472 - val_accuracy: 0.9883
    Epoch 12/30
    263/263 [==============================] - 1s 5ms/step - loss: 0.0068 - accuracy: 0.9976 - val_loss: 0.0556 - val_accuracy: 0.9875
    Epoch 13/30
    263/263 [==============================] - 1s 5ms/step - loss: 0.0059 - accuracy: 0.9981 - val_loss: 0.0474 - val_accuracy: 0.9892
    Epoch 14/30
    263/263 [==============================] - 1s 5ms/step - loss: 0.0058 - accuracy: 0.9979 - val_loss: 0.0555 - val_accuracy: 0.9871
    Epoch 15/30
    263/263 [==============================] - 1s 5ms/step - loss: 0.0052 - accuracy: 0.9983 - val_loss: 0.0749 - val_accuracy: 0.9840
    Epoch 16/30
    263/263 [==============================] - 1s 5ms/step - loss: 0.0083 - accuracy: 0.9972 - val_loss: 0.0551 - val_accuracy: 0.9886
    Epoch 17/30
    263/263 [==============================] - 1s 5ms/step - loss: 0.0030 - accuracy: 0.9992 - val_loss: 0.0513 - val_accuracy: 0.9899
    Epoch 18/30
    263/263 [==============================] - 1s 5ms/step - loss: 0.0038 - accuracy: 0.9989 - val_loss: 0.0616 - val_accuracy: 0.9886
    Epoch 19/30
    263/263 [==============================] - 1s 5ms/step - loss: 0.0058 - accuracy: 0.9979 - val_loss: 0.0589 - val_accuracy: 0.9876



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


![png](mnist_files/mnist_33_0.png)



```
y_pred_base = base_model.predict(test)
y_pred_base = np.argmax(y_pred_base, axis=1)
y_pred_base
```




    array([2, 0, 9, ..., 3, 9, 2])



## Model Tuning

### Data Augmentation

The first step to improving our model is to add data augmentation. We will use ImageDataGenerator, which conveniently allows us to add new training instances that are slightly altered versions of the original training data. This will give us a larger training dataset and help to improve model accuracy. 


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
    263/263 [==============================] - 8s 30ms/step - loss: 0.6454 - accuracy: 0.7822 - val_loss: 0.0845 - val_accuracy: 0.9745
    Epoch 2/30
    263/263 [==============================] - 8s 29ms/step - loss: 0.1414 - accuracy: 0.9563 - val_loss: 0.0687 - val_accuracy: 0.9810
    Epoch 3/30
    263/263 [==============================] - 8s 30ms/step - loss: 0.0942 - accuracy: 0.9709 - val_loss: 0.0433 - val_accuracy: 0.9879
    Epoch 4/30
    263/263 [==============================] - 8s 30ms/step - loss: 0.0764 - accuracy: 0.9773 - val_loss: 0.0541 - val_accuracy: 0.9838
    Epoch 5/30
    263/263 [==============================] - 8s 30ms/step - loss: 0.0681 - accuracy: 0.9790 - val_loss: 0.0365 - val_accuracy: 0.9904
    Epoch 6/30
    263/263 [==============================] - 8s 29ms/step - loss: 0.0594 - accuracy: 0.9822 - val_loss: 0.0342 - val_accuracy: 0.9911
    Epoch 7/30
    263/263 [==============================] - 8s 29ms/step - loss: 0.0552 - accuracy: 0.9842 - val_loss: 0.0249 - val_accuracy: 0.9933
    Epoch 8/30
    263/263 [==============================] - 8s 30ms/step - loss: 0.0539 - accuracy: 0.9841 - val_loss: 0.0254 - val_accuracy: 0.9927
    Epoch 9/30
    263/263 [==============================] - 8s 30ms/step - loss: 0.0478 - accuracy: 0.9855 - val_loss: 0.0345 - val_accuracy: 0.9917
    Epoch 10/30
    263/263 [==============================] - 8s 30ms/step - loss: 0.0437 - accuracy: 0.9868 - val_loss: 0.0314 - val_accuracy: 0.9920
    Epoch 11/30
    263/263 [==============================] - 8s 29ms/step - loss: 0.0432 - accuracy: 0.9869 - val_loss: 0.0334 - val_accuracy: 0.9918
    Epoch 12/30
    263/263 [==============================] - 8s 29ms/step - loss: 0.0455 - accuracy: 0.9865 - val_loss: 0.0211 - val_accuracy: 0.9943
    Epoch 13/30
    263/263 [==============================] - 8s 29ms/step - loss: 0.0432 - accuracy: 0.9882 - val_loss: 0.0485 - val_accuracy: 0.9882
    Epoch 14/30
    263/263 [==============================] - 8s 29ms/step - loss: 0.0401 - accuracy: 0.9882 - val_loss: 0.0294 - val_accuracy: 0.9936
    Epoch 15/30
    263/263 [==============================] - 8s 29ms/step - loss: 0.0411 - accuracy: 0.9876 - val_loss: 0.0295 - val_accuracy: 0.9931
    Epoch 16/30
    263/263 [==============================] - 8s 29ms/step - loss: 0.0386 - accuracy: 0.9888 - val_loss: 0.0296 - val_accuracy: 0.9924
    Epoch 17/30
    263/263 [==============================] - 8s 29ms/step - loss: 0.0423 - accuracy: 0.9883 - val_loss: 0.0261 - val_accuracy: 0.9940
    Epoch 18/30
    263/263 [==============================] - 8s 30ms/step - loss: 0.0402 - accuracy: 0.9889 - val_loss: 0.0302 - val_accuracy: 0.9933
    Epoch 19/30
    263/263 [==============================] - 8s 30ms/step - loss: 0.0398 - accuracy: 0.9885 - val_loss: 0.0278 - val_accuracy: 0.9929
    Epoch 20/30
    263/263 [==============================] - 8s 30ms/step - loss: 0.0413 - accuracy: 0.9889 - val_loss: 0.0249 - val_accuracy: 0.9936
    Epoch 21/30
    263/263 [==============================] - 8s 30ms/step - loss: 0.0410 - accuracy: 0.9890 - val_loss: 0.0276 - val_accuracy: 0.9923
    Epoch 22/30
    263/263 [==============================] - 8s 29ms/step - loss: 0.0410 - accuracy: 0.9883 - val_loss: 0.0244 - val_accuracy: 0.9930



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


![png](mnist_files/mnist_45_0.png)



```
best_score = (max(history.history['val_accuracy']))
print('Best Validation Accuracy: {:.4f}'.format(best_score))
```

    Best Validation Accuracy: 0.9943


The updated CNN is doing quite well now that we have made some adjustments, including stacking two convolutional layers in each step. Let's see if we can take the model a bit further by replacing our early stopping callback with a ReduceLROnPlateau callback, which reduces the learning rate of the model's optimizer when the accuracy begins to plateau.

This callback will allow us to use all of the epochs we scheduled rather than stopping early, and will also potentially increase accuracy once the accuracy begins to plateau (as we saw with earlier executions, the accuracy plateaus before we are done with all 30 epochs).


```
lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=5, factor=0.1)
history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                              epochs=epochs,
                              callbacks=[lr_reduction],
                              validation_data=(X_valid, y_valid))
```


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


![png](mnist_files/mnist_49_0.png)



```
best_score = (max(history.history['val_accuracy']))
print('Best Validation Accuracy: {:.4f}'.format(best_score))
```

    Best Validation Accuracy: 0.9950


Thanks to the ReduceLROnPlateau callback, we were able to make a slight improvement on our model's validation accuracy!

## Hyperopt


```
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials
```


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
    263/263 [==============================] - 8s 30ms/step - loss: 0.5634 - accuracy: 0.8090 - val_loss: 0.0798 - val_accuracy: 0.9752
    Epoch 2/30
    263/263 [==============================] - 7s 28ms/step - loss: 0.1279 - accuracy: 0.9598 - val_loss: 0.0521 - val_accuracy: 0.9839
    Epoch 3/30
    263/263 [==============================] - 8s 29ms/step - loss: 0.0949 - accuracy: 0.9701 - val_loss: 0.0520 - val_accuracy: 0.9846
    Epoch 4/30
    263/263 [==============================] - 7s 28ms/step - loss: 0.0758 - accuracy: 0.9769 - val_loss: 0.0354 - val_accuracy: 0.9895
    Epoch 5/30
    263/263 [==============================] - 7s 28ms/step - loss: 0.0657 - accuracy: 0.9791 - val_loss: 0.0327 - val_accuracy: 0.9904
    Epoch 6/30
    263/263 [==============================] - 7s 28ms/step - loss: 0.0568 - accuracy: 0.9820 - val_loss: 0.0341 - val_accuracy: 0.9901
    Epoch 7/30
    263/263 [==============================] - 7s 28ms/step - loss: 0.0541 - accuracy: 0.9821 - val_loss: 0.0282 - val_accuracy: 0.9917
    Epoch 8/30
    263/263 [==============================] - 7s 28ms/step - loss: 0.0501 - accuracy: 0.9847 - val_loss: 0.0296 - val_accuracy: 0.9923
    Epoch 9/30
    263/263 [==============================] - 7s 28ms/step - loss: 0.0493 - accuracy: 0.9850 - val_loss: 0.0330 - val_accuracy: 0.9910
    Epoch 10/30
    263/263 [==============================] - 7s 29ms/step - loss: 0.0472 - accuracy: 0.9851 - val_loss: 0.0326 - val_accuracy: 0.9913
    Epoch 11/30
    263/263 [==============================] - 7s 28ms/step - loss: 0.0424 - accuracy: 0.9869 - val_loss: 0.0225 - val_accuracy: 0.9933
    Epoch 12/30
    263/263 [==============================] - 7s 28ms/step - loss: 0.0395 - accuracy: 0.9869 - val_loss: 0.0306 - val_accuracy: 0.9923
    Epoch 13/30
    263/263 [==============================] - 8s 29ms/step - loss: 0.0367 - accuracy: 0.9879 - val_loss: 0.0322 - val_accuracy: 0.9914
    Epoch 14/30
    263/263 [==============================] - 7s 28ms/step - loss: 0.0385 - accuracy: 0.9881 - val_loss: 0.0289 - val_accuracy: 0.9918
    Epoch 15/30
    263/263 [==============================] - 7s 28ms/step - loss: 0.0254 - accuracy: 0.9919 - val_loss: 0.0199 - val_accuracy: 0.9948
    Epoch 16/30
    263/263 [==============================] - 7s 28ms/step - loss: 0.0219 - accuracy: 0.9925 - val_loss: 0.0198 - val_accuracy: 0.9950
    Epoch 17/30
    263/263 [==============================] - 7s 28ms/step - loss: 0.0206 - accuracy: 0.9933 - val_loss: 0.0195 - val_accuracy: 0.9948
    Epoch 18/30
    263/263 [==============================] - 7s 28ms/step - loss: 0.0213 - accuracy: 0.9931 - val_loss: 0.0190 - val_accuracy: 0.9952
    Epoch 19/30
    263/263 [==============================] - 7s 28ms/step - loss: 0.0192 - accuracy: 0.9937 - val_loss: 0.0182 - val_accuracy: 0.9955
    Epoch 20/30
    263/263 [==============================] - 7s 28ms/step - loss: 0.0199 - accuracy: 0.9932 - val_loss: 0.0193 - val_accuracy: 0.9951
    Epoch 21/30
    263/263 [==============================] - 7s 28ms/step - loss: 0.0199 - accuracy: 0.9935 - val_loss: 0.0182 - val_accuracy: 0.9949
    Epoch 22/30
    263/263 [==============================] - 7s 28ms/step - loss: 0.0182 - accuracy: 0.9941 - val_loss: 0.0190 - val_accuracy: 0.9949
    Epoch 23/30
    263/263 [==============================] - 7s 28ms/step - loss: 0.0167 - accuracy: 0.9943 - val_loss: 0.0188 - val_accuracy: 0.9952
    Epoch 24/30
    263/263 [==============================] - 8s 29ms/step - loss: 0.0155 - accuracy: 0.9945 - val_loss: 0.0188 - val_accuracy: 0.9951
    Epoch 25/30
    263/263 [==============================] - 8s 29ms/step - loss: 0.0170 - accuracy: 0.9946 - val_loss: 0.0186 - val_accuracy: 0.9951
    Epoch 26/30
    263/263 [==============================] - 8s 30ms/step - loss: 0.0166 - accuracy: 0.9942 - val_loss: 0.0186 - val_accuracy: 0.9951
    Epoch 27/30
    263/263 [==============================] - 8s 29ms/step - loss: 0.0168 - accuracy: 0.9944 - val_loss: 0.0186 - val_accuracy: 0.9951
    Epoch 28/30
    263/263 [==============================] - 7s 28ms/step - loss: 0.0173 - accuracy: 0.9939 - val_loss: 0.0186 - val_accuracy: 0.9949
    Epoch 29/30
    263/263 [==============================] - 7s 28ms/step - loss: 0.0176 - accuracy: 0.9943 - val_loss: 0.0186 - val_accuracy: 0.9949
    Epoch 30/30
    263/263 [==============================] - 7s 28ms/step - loss: 0.0177 - accuracy: 0.9937 - val_loss: 0.0186 - val_accuracy: 0.9949



```
# Save the model
model.save('model.h5')
```


```
model.summary()
```

    Model: "sequential_56"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_333 (Conv2D)          (None, 28, 28, 32)        320       
    _________________________________________________________________
    conv2d_334 (Conv2D)          (None, 28, 28, 32)        9248      
    _________________________________________________________________
    max_pooling2d_168 (MaxPoolin (None, 14, 14, 32)        0         
    _________________________________________________________________
    dropout_220 (Dropout)        (None, 14, 14, 32)        0         
    _________________________________________________________________
    conv2d_335 (Conv2D)          (None, 14, 14, 64)        18496     
    _________________________________________________________________
    conv2d_336 (Conv2D)          (None, 14, 14, 64)        36928     
    _________________________________________________________________
    max_pooling2d_169 (MaxPoolin (None, 7, 7, 64)          0         
    _________________________________________________________________
    dropout_221 (Dropout)        (None, 7, 7, 64)          0         
    _________________________________________________________________
    conv2d_337 (Conv2D)          (None, 7, 7, 128)         73856     
    _________________________________________________________________
    conv2d_338 (Conv2D)          (None, 7, 7, 128)         147584    
    _________________________________________________________________
    max_pooling2d_170 (MaxPoolin (None, 4, 4, 128)         0         
    _________________________________________________________________
    dropout_222 (Dropout)        (None, 4, 4, 128)         0         
    _________________________________________________________________
    flatten_56 (Flatten)         (None, 2048)              0         
    _________________________________________________________________
    dense_112 (Dense)            (None, 512)               1049088   
    _________________________________________________________________
    dropout_223 (Dropout)        (None, 512)               0         
    _________________________________________________________________
    dense_113 (Dense)            (None, 10)                5130      
    =================================================================
    Total params: 1,340,650
    Trainable params: 1,340,650
    Non-trainable params: 0
    _________________________________________________________________



```
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='best_model.png', show_shapes=True, show_layer_names=True)
```




![png](mnist_files/mnist_59_0.png)



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
