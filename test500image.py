import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array


# Test images directory
test_directory = "/Users/bigyansapkota/Desktop/FINALWildlife/test"

# Annotation file path
annotation_file = "/Users/bigyansapkota/Desktop/FINALWildlife/test_annotations.csv"

# Load class indices from file
class_indices = {}
with open("animal_detection_class_indices.txt", "r") as f:
    for line in f:
        class_name, class_index = line.strip().split(":")
        class_indices[class_name] = int(class_index)

# Load the saved model
model = load_model("animal_detection_model.keras")

# Load annotations
annotations = pd.read_csv(annotation_file)

# Initialize variables to keep track of accuracy
correct_predictions = 0
total_images = 0

# Loop through each image in the test directory
for filename in os.listdir(test_directory):
    if filename.endswith(".jpg"):  # Assuming all test images have a .jpg extension
        total_images += 1

        # Check if the filename exists in the annotation file
        if filename in annotations['filename'].values:
            # Construct the full path to the image
            image_path = os.path.join(test_directory, filename)

            # Load and preprocess the image using your existing function
            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Use the model to predict the class probabilities
            predictions = model.predict(img_array)

            # Get the predicted class label
            predicted_class_index = np.argmax(predictions)
            predicted_class_label = [k for k, v in class_indices.items() if v == predicted_class_index][0]

            # Extract the true class label from the annotation file
            true_class_label = annotations[annotations['filename'] == filename]['class'].values[0]

            # Check if the prediction is correct
            if predicted_class_label == true_class_label:
                correct_predictions += 1

            # Optionally, you can display the input image along with the predicted class
            # plt.imshow(img)
            # plt.axis('off')
            # plt.title(f"Predicted Class: {predicted_class_label}")
            # plt.show()

# Calculate and print the accuracy
accuracy = correct_predictions / total_images
print(f"Accuracy: {accuracy * 100:.2f}%")
