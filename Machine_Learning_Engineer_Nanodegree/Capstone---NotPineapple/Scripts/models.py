# import necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

# creating a fully connected model
class FullConn:
    # build model function
    @staticmethod
    def build_model(width, height, depth, nclasses):
        # initializing model
        model = Sequential()
        input_shape = (width, height, depth)

        # first set of FLATTEN -> FC -> RELU layer
        model.add(Flatten(data_format='channels_last', input_shape=input_shape))
        model.add(Dense(1200))
        model.add(Activation('relu'))

        # second set of FC -> RELU layer
        model.add(Dense(800))
        model.add(Activation('relu'))

        # third set of FC -> RELU layer
        model.add(Dense(400))
        model.add(Activation('relu'))

        # softmax classifier
        model.add(Dense(nclasses))
        model.add(Activation('softmax'))

        # return the constructed architecture
        return model



# creating cnn model based on LeNet architechture
class LeNet:
    # build model function
    @staticmethod
    def build_model(width, height, depth, nclasses):
        # initializing model
        model = Sequential()
        input_shape = (width, height, depth)

        # first set of CONV -> RELU -> MAXPOOL Layers
        model.add(Conv2D(filters=20, kernel_size=(5,5), strides=(1,1), \
                         padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2),\
                               padding='same'))

        # second set of CONV -> RELU -> MAXPOOL Layers
        model.add(Conv2D(filters=50, kernel_size=(5,5), strides=(1,1), \
                         padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2),\
                               padding='same'))

        # one set of FC -> RELU layers
        model.add(Flatten(data_format='channels_last'))
        model.add(Dense(500))
        model.add(Activation('relu'))

        # softmax classifier
        model.add(Dense(nclasses))
        model.add(Activation('softmax'))

        # return the constructed architechture
        return model



# creating cnn model based on custom architechture
class CustomConv:
    # build model function
    @staticmethod
    def build_model(width, height, depth, nclasses):
        # initializing model
        model = Sequential()
        input_shape = (width, height, depth)

        # first set of CONV -> RELU -> MAXPOOL Layers
        model.add(Conv2D(filters=20, kernel_size=(5,5), strides=(1,1), \
                         padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2),\
        #                        padding='same'))

        # # second set of CONV -> RELU -> MAXPOOL Layers
        # model.add(Conv2D(filters=50, kernel_size=(5,5), strides=(1,1), \
        #                  padding='same', input_shape=input_shape))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2),\
        #                        padding='same'))

        # one set of FC -> RELU layers
        model.add(Flatten(data_format='channels_last'))
        model.add(Dense(500))
        model.add(Activation('relu'))

        # softmax classifier
        model.add(Dense(nclasses))
        model.add(Activation('softmax'))

        # return the constructed architechture
        return model