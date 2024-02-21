import os
import cv2
import numpy as np
import pandas as pd
from collections import Counter
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

random.seed(29)

# Set random seed for NumPy
np.random.seed(29)

# Function to read the class labels from the CSV file
def read_class_labels_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path)
    class_labels = {}
    for index, row in df.iterrows():
        image_name = row['Image Index']
        finding_label = row['Finding Labels']
        class_labels[image_name] = finding_label
    return class_labels


# Function to assign class labels to images in multiple folders
def assign_class_to_images(folders, class_labels):
    assigned_images = []
    class_counter = Counter()
    for folder in folders:
        for filename in os.listdir(folder):
            image_path = os.path.join(folder, filename)
            if os.path.isfile(image_path):
                image_name = os.path.basename(image_path)
                if image_name in class_labels:
                    class_label = class_labels[image_name]
                    assigned_images.append([image_path, class_label])
                    class_counter[class_label] += 1

    # Filter out images in the top 15 classes.
    top_n = 15
    top_classes = [class_label for class_label, count in class_counter.most_common(top_n)]
    assigned_images = [img_info for img_info in assigned_images if img_info[1] in top_classes]

    print(top_classes)

    return assigned_images

folders = []

# Assigning the 12 folders to the list.
for i in range(1, 13):
    folder_path = f'NIH Kaggle Data/images_{str(i).zfill(3)}/images'
    folders.append(folder_path)

# CSV file path
csv_file_path = 'NIH Kaggle Data/Data_Entry_2017.csv'

# Load class labels from the CSV file
class_labels = read_class_labels_from_csv(csv_file_path)


# Assign class labels to the images in the folders
assigned_images = assign_class_to_images(folders, class_labels)

print("Number of assigned images:", len(assigned_images))

# Separate the images and class labels
images = []
labels = []
for img_path, class_label in assigned_images:
    img_arr = cv2.imread(img_path)
    img_arr_resized = cv2.resize(img_arr, (224, 224))
    images.append(img_arr_resized)
    labels.append(class_label)

class_names = ['No Finding', 'Infiltration', 'Atelectasis', 'Effusion', 'Nodule', 'Pneumothorax', 'Mass',
               'Effusion|Infiltration','Atelectasis|Infiltration', 'Consolidation', 'Atelectasis|Effusion',
               'Pleural_Thickening', 'Cardiomegaly','Emphysema', 'Infiltration|Nodule']

# Create a set to keep track of the labels we've already seen
seen_labels = set()

# Create a list to store the example images
example_images = []

# Traverse the labels to find unique ones and get their corresponding images
for i, label in enumerate(labels):
    if label not in seen_labels:
        seen_labels.add(label)
        example_images.append(images[i])
    if len(seen_labels) == len(class_names):
        break

# Plot the images using matplotlib
fig, axes = plt.subplots(5, 3, figsize=(15, 15))

for i, (img, class_name) in enumerate(zip(example_images, class_names)):
    row = i // 3
    col = i % 3
    axes[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Convert from BGR to RGB
    axes[row, col].set_title(class_name)
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
