from tensorflow import keras
import tensorflow
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D,Input,Concatenate,Activation
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.vgg16 import VGG16
from keras.layers import concatenate

def create_dms_model_v0_1(img_rows,img_cols,color_type):

    model = Sequential()

    ## CNN 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, color_type)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    ## CNN 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    ## CNN 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))

    ## Output
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))

    return model



def fire_module(x, squeeze_filters, expand_filters):
    squeeze = Conv2D(squeeze_filters, (1, 1), activation='relu', padding='valid')(x)
    expand_1x1 = Conv2D(expand_filters, (1, 1), activation='relu', padding='valid')(squeeze)
    expand_3x3 = Conv2D(expand_filters, (3, 3), activation='relu', padding='same')(squeeze)
    return concatenate([expand_1x1, expand_3x3], axis=-1)

def create_squeezenet(input_shape=(224, 224, 3), num_classes=10):
    """ Keras Implementation of SqueezeNet(arXiv 1602.07360)

       @param nb_classes: total number of final categories

       Arguments:
       inputs -- shape of the input images (channel, cols, rows)

       """
    input_tensor = Input(shape=input_shape)

    x = Conv2D(96, (7, 7), strides=(2, 2), activation='relu', padding='valid')(input_tensor)
    x = BatchNormalization(momentum=0.9)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)

    x = fire_module(x, squeeze_filters=16, expand_filters=64)
    x = BatchNormalization(momentum=0.9)(x)
    x = fire_module(x, squeeze_filters=16, expand_filters=64)
    x = BatchNormalization(momentum=0.9)(x)
    x = fire_module(x, squeeze_filters=32, expand_filters=128)
    x = BatchNormalization(momentum=0.9)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)

    x = fire_module(x, squeeze_filters=32, expand_filters=128)
    x = BatchNormalization(momentum=0.9)(x)
    x = fire_module(x, squeeze_filters=48, expand_filters=192)
    x = BatchNormalization(momentum=0.9)(x)
    x = fire_module(x, squeeze_filters=48, expand_filters=192)
    x = BatchNormalization(momentum=0.9)(x)
    x = fire_module(x, squeeze_filters=64, expand_filters=256)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)

    x = fire_module(x, squeeze_filters=64, expand_filters=256)
    x = Dropout(0.5)(x)

    x = Conv2D(num_classes, (1, 1), activation='relu', padding='valid')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = GlobalAveragePooling2D()(x)
    output_tensor = Activation('softmax')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model
