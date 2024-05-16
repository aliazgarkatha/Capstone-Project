import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array


#Getting the test image path
image_path = r"E:\AIDA\Semester 2\Capstone project\Data sets\Final dataset\Wildlife Security System.v2i.tensorflow\testing pic\test_images\monkey.png"


#Getting class indices from previously saved file
class_indices = {}
with open("animal_detection_class_indices.txt", "r") as f:
    for line in f:
        class_name, class_index = line.strip().split(":")
        class_indices[class_name] = int(class_index)


# Loading our saved model
model = load_model("animal_detection_model.keras")


def test_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Model predicting probabilities of classes
    predictions = model.predict(img_array)

    # Getting predicted class with maximum probability
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = [k for k, v in class_indices.items() if v == predicted_class_index][0]

    # Displaying imput image and the predicted class
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted Class: {predicted_class_label}")
    plt.show()

# Calling test_image function to test image
test_image(image_path)
