import os
import tensorflow as tf
import datetime
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 3 = INFO, WARNING, and ERROR
from tqdm import tqdm
import numpy as np
import pandas as pd
from IPython.display import FileLink
import sys
#from keras.models import load_model        #----SK - Loads only hd5 model
from tensorflow.keras.models import load_model      #----SK - Loads only h5 model
import warnings
warnings.filterwarnings('ignore')
from data_analysis import *
from model import *
from data_augmentation import *
from Inferance import *


def get_available_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
def train():
    print("******DMS V0.1******")
    get_available_gpus()
    dataset = load_data()
    img_rows = 64#64  # dimension of images
    img_cols = 64#64
    color_type = 3  # grey
    nb_test_samples = 200
    batch_size = 20  # 40
    nb_epoch = 6  # 10
    # loading train images
    #x_train, x_test, y_train, y_test = read_and_normalize_train_data(img_rows, img_cols, color_type)

    # loading validation images
    test_files, test_targets = read_and_normalize_sampled_test_data(nb_test_samples, img_rows, img_cols, color_type)
    #analyse_dataset(x_train, x_test)
    models_dir = "../data"
    # if not os.path.exists(models_dir):
    #     os.makedirs(models_dir)

    # checkpointer = ModelCheckpoint(filepath=models_dir+'/dms_model_v0.1.keras',
    #                                monitor='val_loss', mode='min',
    #                                save_weights_only=True,
    #                                verbose=1, save_best_only=True)
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
    model = create_dms_model_v0_1(img_rows,img_cols,color_type)
    model.summary()
    # Compiling the model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # # Training
    # # Number of batch size and epochs
    #
    # history1 = model.fit(x_train, y_train,
    #                     validation_data=(x_test, y_test),
    #                     callbacks=[checkpointer],
    #                     epochs=nb_epoch, batch_size=batch_size, verbose=1)
    #
    # # model.load_weights('saved_models/weights_best_vanilla.hdf5')
    # print('History of the training', history1.history)
    #
    # #Evaluate Model
    # score1 = model.evaluate(x_test, y_test, verbose=1)
    #
    # print('Loss: ', score1[0])
    # print('Accuracy: ', score1[1] * 100, ' %')

    # With Data Augumentation

    train_datagen,test_datagen = augment_training_data()
    # nb_train_samples = x_train.shape[0]
    # nb_validation_samples = x_test.shape[0]
    # training_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    # validation_generator = test_datagen.flow(x_test, y_test, batch_size=batch_size)
    #
    # checkpointer2 = ModelCheckpoint(filepath=models_dir + '/dms_model_v0.1_with_Augmentation.keras',
    #                                monitor='val_loss', mode='min',
    #                                save_weights_only=True,
    #                                verbose=1, save_best_only=True)
    # history_v2 = model.fit_generator(training_generator,
    #                                  steps_per_epoch=nb_train_samples // batch_size,
    #                                  epochs=nb_epoch,
    #                                  verbose=1,
    #                                  validation_data=validation_generator,
    #                                  callbacks=[checkpointer2],
    #                                  validation_steps=nb_validation_samples // batch_size)
    #
    # # Evaluate and compare the performance of the new model
    # score2 = model.evaluate_generator(validation_generator, nb_validation_samples // batch_size)
    # #print("Loss for model 1", score1[0])
    # print("Loss for model 2 (data augmentation):", score2[0])
    #
    # #print("Test accuracy for model 1", score1[1])
    # print("Test accuracy for model 2 (data augmentation):", score2[1])
    # model.save("../data/dms_model_v0.1_04_01_2024_aug.keras")

    #-----------------------------------------------------------------------------------------------------------------

    # Squeezenet Model

    #----------------------------------------------------------------------------------------------------------------
    print("##########-SqueezeDet Training-############")
    # Create a tf.distribute.Strategy object
    strategy = tensorflow.distribute.MirroredStrategy()
    GLOBAL_BATCH_SIZE = strategy.num_replicas_in_sync * batch_size
    print(f'Number of devices: {strategy.num_replicas_in_sync}')
    img_rows = 224
    img_cols = 224
    x_trainSq, x_testSq, y_trainSq, y_testSq = read_and_normalize_train_data(img_rows, img_cols, color_type)
    # loading validation images
    test_files, test_targets = read_and_normalize_sampled_test_data(nb_test_samples, img_rows, img_cols, color_type)

    nb_train_samples = x_trainSq.shape[0]
    nb_validation_samples = x_testSq.shape[0]
    training_generator = train_datagen.flow(x_trainSq, y_trainSq, batch_size=batch_size)
    validation_generator = test_datagen.flow(x_testSq, y_testSq, batch_size=batch_size)

    #SqueezeNet_model = SqueezeNet(NUMBER_CLASSES, inputs=(3, img_rows, img_cols))
    with strategy.scope():
        SqueezeNet_model = create_squeezenet()
        # Compiling the model
        SqueezeNet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(SqueezeNet_model.inputs)
        checkpointer3 = ModelCheckpoint(filepath=models_dir + '/dms_Squeezenet_v1.1_with_Augmentation_weights.h5',
                                        monitor='val_loss', mode='min',
                                        save_weights_only=True,
                                        verbose=1, save_best_only=True)
        history_SqueezNet = SqueezeNet_model.fit_generator(training_generator,
                                         steps_per_epoch=nb_train_samples // batch_size,
                                         epochs=nb_epoch,
                                         verbose=1,
                                         validation_data=validation_generator,
                                         callbacks=[checkpointer3],
                                         validation_steps=nb_validation_samples // batch_size)

    #SqueezeNet_model.save_weights('../data/dms_v0.1_squeezenet_weights.hd5', overwrite=True)
    SqueezeNet_model.save('../data/dms_v1.1_squeezenet.h5',overwrite=True,save_format='tf')
    model_parms = {'nb_class': NUMBER_CLASSES,
                   'nb_train_samples': nb_train_samples,
                   'nb_val_samples': nb_validation_samples,
                   'classes': NUMBER_CLASSES,
                   'channels': 3,
                   'height': img_rows,
                   'width': img_cols}
    write_json(model_parms, fname='../data/SqueezeNet_parms.json')
    plot_train_history(history_SqueezNet)

#infer /home/saurabh/Project/DMS/DriverMonitoringSystem/data/dms_Squeezenet_v0.1_with_Augmentation.h5 /home/saurabh/Project/DMS/KaggleData/imgs/img_1.jpg

if __name__ == '__main__':
    print(f'\nTensorflow version = {tf.__version__}\n')
    print(f'\n{tf.config.list_physical_devices("GPU")}\n')
    print("\n*******************DMS Application v1.1***************************\n")
    # if len(sys.argv) != 4:
    #     print("\n[ERROR] Missing Arguments :-hint:(Usage: python main.py <train or infer>)\n")
    #     sys.exit(1)
    option = sys.argv[1].lower()
    if option == "train":
        train()
    elif option == "infer":
        if len(sys.argv) != 4:
            print("[ERROR] Missing Arguments :-hint:(Usage: python main.py infer <model_path> <image_path>)")
            sys.exit(1)
        model_path = sys.argv[2]
        image_path = sys.argv[3]
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found.")
            sys.exit(1)
        print(f"\n[Infer]: Model Path Selected {model_path}")
        print(f"\n[Infer]: Image Path Selected {image_path}")
        # Load the pre-trained SqueezeNet model
        try:
            # Load the pre-trained SqueezeNet model
            model = load_model(model_path)
        except Exception as e:
            print(f"Error loading the model: {e}")
            sys.exit(1)
        #model = load_model(model_path)
        predicted_class, confidence = predict_image_class(model, image_path, labels)
        print(f"\nPredicted class: {predicted_class} --> {activity_map[predicted_class]}")
        print(f"\nConfidence: {confidence}")

    #print("Invalid option. Use 'train' or 'infer'.")

#https://www.kaggle.com/code/pierrelouisdanieau/computer-vision-tips-to-increase-accuracy