import tensorflow








import os
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten
from tensorflow.keras.models import load_model,Sequential
from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing import image
import numpy as np
import random,shutil
import matplotlib.pyplot as plt

img = image.load_img('/content/dataset_new/test/yawn/100.jpg', target_size=(24, 24,1))

img.show()
x = image.img_to_array(img)
print(type(x))
print(x.shape)
plt.imshow(x/255.)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)

train_data =train_datagen.flow_from_directory('/content/dataset_new/train',shuffle=True,batch_size=16,color_mode='grayscale',target_size=(24,24),class_mode='categorical')
valid_data =test_datagen.flow_from_directory('/content/dataset_new/test',shuffle=True,batch_size=16,color_mode='grayscale',target_size=(24,24),class_mode='categorical')

SPE= len(train_data.classes)//16
VS = len(valid_data.classes)//16
print(SPE,VS)

train_data

model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(24,24,1)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit_generator(train_data, validation_data=valid_data,epochs=25,steps_per_epoch=SPE ,validation_steps=VS,verbose=1)

model.save('drowsiness.h5')