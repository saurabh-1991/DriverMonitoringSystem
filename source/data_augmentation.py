from keras.preprocessing.image import ImageDataGenerator

def augment_training_data():
    trainDatagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.10,
        brightness_range=[0.6, 1.4],
        channel_shift_range=0.7,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        horizontal_flip=True,
        validation_split=0.2,
        fill_mode='nearest')
    testDatagen = ImageDataGenerator(rescale=1.0/ 255, validation_split = 0.2)
    return trainDatagen,testDatagen

