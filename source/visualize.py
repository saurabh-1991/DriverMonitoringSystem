import cv2
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