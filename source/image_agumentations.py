import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims


def plot_augimages(paths, datagen):
    plt.figure(figsize=(14, 28))
    plt.suptitle('Augmented Images')

    midx = 0
    for path in paths:
        data = Image.open(path)
        data = data.resize((224, 224))
        samples = expand_dims(data, 0)
        it = datagen.flow(samples, batch_size=1)

        # Show Original Image
        plt.subplot(10, 5, midx + 1)
        plt.imshow(data)
        plt.axis('off')

        # Show Augmented Images
        for idx, i in enumerate(range(4)):
            midx += 1
            plt.subplot(10, 5, midx + 1)

            batch = it.next()
            image = batch[0].astype('uint8')
            plt.imshow(image)
            plt.axis('off')
        midx += 1

    plt.tight_layout()
    plt.show()


def augment_training_data():
    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.10,
        brightness_range=[0.6, 1.4],
        channel_shift_range=0.7,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest')
    return datagen
