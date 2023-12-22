from keras.callbacks import ReduceLROnPlateau
from keras.applications.vgg19 import VGG19
from keras.applications import ResNet50
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.python.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Flatten
from sklearn.model_selection import train_test_split
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50


def generate_tfl_vgg19_model(train_data):
    y_count = len(train_data['classname'].unique())
    # include_top = False means that we doesnt include fully connected top layer we will add them accordingly
    vgg19 = VGG19(include_top=False, input_shape=(560, 560, 3), weights='imagenet')

    # training of all the convolution is set to false
    for layer in vgg19.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(vgg19.output)
    predictions = Dense(y_count, activation='softmax')(x)

    model_vgg19 = Model(inputs=vgg19.input, outputs=predictions)
    return model_vgg19


def generate_tfl_resnet50_model(train_data):
    y_count = len(train_data['classname'].unique())
    # include_top = False means that we doesnt include fully connected top layer we will add them accordingly
    resNet50 = ResNet50(include_top=False, input_shape=(560, 560, 3), weights='imagenet')

    # training of all the convolution is set to false
    for layer in resNet50.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(resNet50.output)
    predictions = Dense(y_count, activation='softmax')(x)

    model_resNet50 = Model(inputs=resNet50.input, outputs=predictions)
    return model_resNet50


def compile_tfl_vgg19_model(vgg19_model):
    vgg19_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    rlrp_vgg19 = ReduceLROnPlateau(monitor="val_loss", factor=0.01, patience=2, verbose=2,
                                   mode="auto", min_delta=0.0001, cooldown=0, min_lr=0)
    print(vgg19_model.summary())
    return rlrp_vgg19


def compile_tfl_resnet50_model(resnet50_model):
    resnet50_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    rlrp_resNet50 = ReduceLROnPlateau(monitor="val_loss", factor=0.01, patience=2, verbose=2,
                                      mode="auto", min_delta=0.0001, cooldown=0, min_lr=0)
    print(resnet50_model.summary())
    return rlrp_resNet50


def prepare_train_test_data(train_data):
    X, Y = train_data[['path', 'classname']], train_data['classname']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test

def train_generator_vgg19(X_train,datagen):
    train_generator_vgg_19 = datagen.flow_from_dataframe(
    X_train,  # This is the source directory for training images
    x_col='path',
    y_col='classname',
    target_size=(560, 560),  # All images will be resized to 150x150
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    preprocessing_function=preprocess_input_vgg19
    )
    return train_generator_vgg_19
def train_generator_resnet50(X_train,datagen):
    train_generator_resnet50 = datagen.flow_from_dataframe(
        X_train,  # This is the source directory for training images
        x_col='path',
        y_col='classname',
        target_size=(560, 560),  # All images will be resized to 150x150
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        preprocessing_function=preprocess_input_resnet50)
    return train_generator_resnet50
def train_generator_dms_custom_model1(X_train,datagen):
    train_generator_dms_model1 = datagen.flow_from_dataframe(
    X_train,  # This is the source directory for training images
    x_col='path',
    y_col='classname',
    target_size=(64, 64),  # All images will be resized to 150x150
    batch_size=40,
    class_mode="categorical",
    shuffle=True,
    )
    return train_generator_dms_model1

def val_generator_vgg19(X_test,datagen):
    val_generator_vgg_19 = datagen.flow_from_dataframe(
        X_test,  # This is the source directory for training images
        x_col='path',
        y_col='classname',
        target_size=(560, 560),  # All images will be resized to 150x150
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        preprocessing_function=preprocess_input_vgg19
    )
    return val_generator_vgg_19
def val_generator_resnet50(X_test,datagen):
    val_generator_resnet50 = datagen.flow_from_dataframe(
    X_test,  # This is the source directory for training images
    x_col='path',
    y_col='classname',
    target_size=(560, 560),  # All images will be resized to 150x150
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    preprocessing_function=preprocess_input_resnet50
    )

def val_generator_dms_model1(X_test,datagen):
    val_generator_custom_model = datagen.flow_from_dataframe(
    X_test,  # This is the source directory for training images
    x_col='path',
    y_col='classname',
    target_size=(64, 64),  # All images will be resized to 150x150
    batch_size=40,
    class_mode="categorical",
    shuffle=True,
    )
    return val_generator_custom_model
def create_dms_custom_model1():
    model = Sequential()
    ## CNN 1
    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(BatchNormalization())
    model.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))
    ## CNN 2
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))
    ## CNN 3
    model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
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
