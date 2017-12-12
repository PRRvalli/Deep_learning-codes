
# coding: utf-8

# In[15]:

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, Flatten
from keras.optimizers import RMSprop, adam
from keras.models import Model
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D


# In[6]:

def loaddata():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return (x_train, y_train), (x_test, y_test)


# In[8]:

(x_train, y_train), (x_test, y_test) = loaddata()


# In[14]:

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)


# In[16]:

# model 
input_img = Input(shape=(32, 32, 3))

conv_1=Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
conv_2=Conv2D(32, (3, 3), activation='relu', padding='same')(conv_1)
max_pool_1=MaxPooling2D((2, 2), padding='same')(conv_2)
drop_1=Dropout(0.25)(max_pool_1)


conv_3=Conv2D(32, (3, 3), activation='relu', padding='same')(drop_1)
conv_4=Conv2D(32, (3, 3), activation='relu', padding='same')(conv_3)
max_pool_2=MaxPooling2D((2, 2), padding='same')(conv_4)
drop_2=Dropout(0.25)(max_pool_2)

flat=Flatten()(drop_2)
dense_1=Dense(512, activation='relu')(flat)
drop_1=Dropout(0.2)(dense_1)
out=Dense(10, activation='softmax')(drop_1)

CNN=Model(input_img,out)


# In[17]:

CNN.summary()


# In[ ]:

CNN.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = CNN.fit(x_train, y_train,
                    batch_size=128,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_test, y_test))


# In[ ]:

scores = CNN.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

