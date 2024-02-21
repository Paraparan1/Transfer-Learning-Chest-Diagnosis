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

    final_assigned_images = []
    for top_class in top_classes:
        class_images = [img_info for img_info in assigned_images if img_info[1] == top_class]
        final_assigned_images.extend(class_images[:800])

    return final_assigned_images
    print(top_classes)

    return final_assigned_images

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

# Convert images and labels to NumPy arrays
images = np.array(images).reshape(-1, 224, 224, 3)
labels = np.array(labels)


# Data split of: Train 60%, Test 20%, Validation 20% and stratify is used to make sure all unique labels are present.
X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.2, random_state=3, stratify=labels)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=3)


# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Fit the encoder on the combined class labels
label_encoder.fit(list(Y_train) + list(Y_val) + list(Y_test))

Y_train = label_encoder.transform(Y_train)
Y_val = label_encoder.transform(Y_val)
Y_test = label_encoder.transform(Y_test)

# Check Y_train
print("Sample Y_train values:", Y_train[:10])  # Print the first 10 values of Y_train

# Check Y_val
print("Sample Y_val values:", Y_val[:10])  # Print the first 10 values of Y_val

# Check Y_test
print("Sample Y_test values:", Y_test[:10])  # Print the first 10 values of Y_test

save_dir = 'Processed Data/class_balanced_NIH2'
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
np.save(os.path.join(save_dir, 'Y_train.npy'), Y_train)
np.save(os.path.join(save_dir, 'Y_val.npy'), Y_val)
np.save(os.path.join(save_dir, 'Y_test.npy'), Y_test)

# Plot distibution
# # Count occurrences of each class label for each set
# class_counts_train = {}
# class_counts_val = {}
# class_counts_test = {}
#
# for class_label in unique_classes:
#     class_counts_train[class_label] = np.sum(Y_train == class_label)
#     class_counts_val[class_label] = np.sum(Y_val == class_label)
#     class_counts_test[class_label] = np.sum(Y_test == class_label)
#
# print(class_counts_train)
# print(class_counts_val)
# print(class_counts_test)
#
# try:
#     # Create a bar plot for the train set class distribution
#     plt.figure(figsize=(12, 6))
#     plt.bar(class_counts_train.keys(), class_counts_train.values())
#     plt.xlabel('Class Labels')
#     plt.ylabel('Number of Images')
#     plt.title('Train Set Class Distribution')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#
#     # Annotate the bars with the number of images
#     for class_label, count in class_counts_train.items():
#         plt.text(class_label, count + 5, str(count), ha='center', va='bottom')
#
#     # Save the train set plot as an image file
#     train_plot_save_path = 'balanced_train_class_distribution_plot.png'
#     plt.savefig(train_plot_save_path)
#
#     # Display the train set plot
#     plt.show()
#
#     # Create a bar plot for the validation set class distribution
#     plt.figure(figsize=(12, 6))
#     plt.bar(class_counts_val.keys(), class_counts_val.values())
#     plt.xlabel('Class Labels')
#     plt.ylabel('Number of Images')
#     plt.title('Validation Set Class Distribution')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#
#     # Annotate the bars with the number of images
#     for class_label, count in class_counts_val.items():
#         plt.text(class_label, count + 5, str(count), ha='center', va='bottom')
#
#     # Save the validation set plot as an image file
#     val_plot_save_path = 'balanced_validation_class_distribution_plot.png'
#     plt.savefig(val_plot_save_path)
#
#     # Display the validation set plot
#     plt.show()
#
#     # Create a bar plot for the test set class distribution
#     plt.figure(figsize=(12, 6))
#     plt.bar(class_counts_test.keys(), class_counts_test.values())
#     plt.xlabel('Class Labels')
#     plt.ylabel('Number of Images')
#     plt.title('Test Set Class Distribution')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#
#     # Annotate the bars with the number of images
#     for class_label, count in class_counts_test.items():
#         plt.text(class_label, count + 5, str(count), ha='center', va='bottom')
#
#     # Save the test set plot as an image file
#     test_plot_save_path = 'balanced_test_class_distribution_plot.png'
#     plt.savefig(test_plot_save_path)
#
#     # Display the test set plot
#     plt.show()
# except (ValueError, RuntimeError) as e:
#     print(e)


