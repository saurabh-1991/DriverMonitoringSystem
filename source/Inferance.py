import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input


labels = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8']
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def predict_image_class(model, image_path, labels):
    preprocessed_image = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]
    class_label = labels[predicted_class]
    return class_label, confidence