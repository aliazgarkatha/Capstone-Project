import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report

# Accessing all test images
test_directory = "/Users/bigyansapkota/Desktop/FINALWildlife/test"

# Annotation file path
annotation_file = "/Users/bigyansapkota/Desktop/FINALWildlife/test_annotations.csv"

# Opening class indices file
class_indices = {}
with open("animal_detection_class_indices.txt", "r") as f:
    for line in f:
        class_name, class_index = line.strip().split(":")
        class_indices[class_name] = int(class_index)

# Loading our saved model
model = load_model("animal_detection_model.keras")

# Load annotations 
annotations = pd.read_csv(annotation_file)

# Lists to store true class labels and predicted class labels
true_class_labels = []
predicted_class_labels = []

# Going through each image on the folder with the filename
for filename in os.listdir(test_directory):
    if filename.endswith(".jpg"):
        if filename in annotations['filename'].values:
            # making the full image path
            image_path = os.path.join(test_directory, filename)

            # Loading and preprocessing the image
            image = load_img(image_path, target_size=(224, 224))
            image_array = img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = image_array / 255.0

            # Getting the predicted class from our model
            model_predictions = model.predict(image_array)
            predicted_class_index = np.argmax(model_predictions)
            predicted_class = [k for k, v in class_indices.items() if v == predicted_class_index][0]

            # Getting true class label from our annotation file
            true_class_label = annotations[annotations['filename'] == filename]['class'].values[0]

            # Filling the true and predicted class labels in the lists
            true_class_labels.append(true_class_label)
            predicted_class_labels.append(predicted_class)

# Converting labels to numerical values
true_labels_numeric = [class_indices[label] for label in true_class_labels]
predicted_labels_numeric = [class_indices[label] for label in predicted_class_labels]

# classification report and confusion matrix
classification_report = classification_report(true_labels_numeric, predicted_labels_numeric, target_names=class_indices.keys())
conf_matrix = confusion_matrix(true_labels_numeric, predicted_labels_numeric)

print("\nClassification Report:")
print(classification_report)

print("\nConfusion Matrix:")
print(conf_matrix)

# Plotting confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Reds)
plt.title('Animal Detection:Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_indices.keys()))
plt.xticks(tick_marks, class_indices.keys(), rotation=50)
plt.yticks(tick_marks, class_indices.keys())
plt.xlabel('Predicted_Class_Label')
plt.ylabel('True_Class_Label')
plt.show()
