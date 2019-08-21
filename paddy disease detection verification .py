# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:15:20 2019

@author: ramle
"""

# Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(units = 1024, activation = 'relu'))

classifier.add(Dense(units = 3, activation = 'softmax'))

# Compiling the CNN
from keras.optimizers import RMSprop, SGD
#classifier.compile(loss = 'categorical_crossentropy',
#              optimizer = RMSprop(lr = 0.001),
#              metrics = ['accuracy'])

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('D:/project verify/Train',
                                                 target_size = (128, 128),
                                                 batch_size = 16,
                                                 class_mode = 'categorical',
                                                 shuffle=True)

test_set = test_datagen.flow_from_directory('D:/project verify/Test',
                                            target_size = (128, 128),
                                            batch_size = 16,
                                            class_mode = 'categorical',
                                            shuffle=False)
nb_train_samples=48905
nb_validation_samples=16421
batch_size=16

classifier.fit_generator(training_set,
                         steps_per_epoch = nb_train_samples // batch_size,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = nb_validation_samples // batch_size)


#from keras.models import load_model

classifier.save('plant_disease.h5') # creates a HDF5 file ‘my_model.h5’
#del model # deletes the existing model



