import cv2
import os
from Inferance import *
from source.data_analysis import activity_map

a = []

def load_img(path):
    for img in os.listdir(path):
        a.append(img)
    return a

def read_img(path):
    image = cv2.imread(path)
    return image


def show_img(image):
    cv2.imshow("Image",image)
    cv2.waitKey(0)


def overlay_text(image,predicted_class):
    # cv2.putText(image, "Label: {}, {:.2f}%".format(predicted_class, confidence * 100),
    #         (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(image, f"{predicted_class}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


# def iterate(list_a):
#     # while True:
#     for i in list_a:
#         tell = input("Press 'y' for next prediction or 'n' to terminate: ")
#         image_path = "C:/Users/shralatt/Desktop/Project/VGG19_project/Dataset/imgs/test/"
#         test = image_path + i
#         if tell == 'y':
#             predicted_class, confidence = predict_image_class(model, test, labels)
#             print(f"\nPredicted class: {predicted_class} --> {activity_map[predicted_class]}")
#             print(f"\nConfidence: {confidence}")
#             image = read_img(test)
#             overlay_text(image, activity_map[predicted_class])
#             show_img(image)
#
#         elif tell == 'n':
#             break
#
#         else:
#             print("Please enter valid command 'y' or 'n': ")