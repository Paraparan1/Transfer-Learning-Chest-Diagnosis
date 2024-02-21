import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_dir = 'Tuberculosis Kaggle Data/archive (5)/TB_Chest_Radiography_Database'

class_names = ['Tuberculosis', 'Normal']

processed_data = []

def preprocess_data():
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        class_num = class_names.index(class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img_arr = cv2.imread(img_path)
            try:
                img_arr_resized = cv2.resize(img_arr, (224, 224))
            except Exception as e:
                print(img_path)
            processed_data.append([img_arr_resized, class_num])
    return processed_data

preprocess_data()

images = [img for img, lab in processed_data]
labels = [lab for img, lab in processed_data]

# Reshaping the array to original dimensions
images = np.array(images).reshape(-1, 224, 224, 3)
labels = np.array(labels)

# Data split of: Train 60%, Test 20%, Validation 20%
X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.2, random_state=3)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=3)

# all_classes = np.concatenate((Y_train, Y_val, Y_test))
#
# # Get the unique class labels for all sets
# unique_classes = np.unique(all_classes)
#
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
# class_label_to_name = {i: name for i, name in enumerate(class_names)}
#
#
# try:
#     # Create a bar plot for the train set class distribution
#     plt.figure(figsize=(6, 5))
#     plt.bar(class_counts_train.keys(), class_counts_train.values())
#     plt.xlabel('Class Labels')
#     plt.ylabel('Number of Images')
#     plt.title('Train Set Class Distribution')
#     plt.xticks(list(class_label_to_name.keys()), list(class_label_to_name.values()))
#     plt.tight_layout()
#
#     # Annotate the bars with the number of images
#     for class_label, count in class_counts_train.items():
#         plt.text(class_label, count + 5, str(count), ha='center', va='bottom')
#
#     # Save the train set plot as an image file
#     train_plot_save_path = 'train_class_distribution_plot.png'
#     plt.savefig(train_plot_save_path)
#
#     # Display the train set plot
#     plt.show()
#
#     # Create a bar plot for the validation set class distribution
#     plt.figure(figsize=(6, 5))
#     plt.bar(class_counts_val.keys(), class_counts_val.values())
#     plt.xlabel('Class Labels')
#     plt.ylabel('Number of Images')
#     plt.title('Validation Set Class Distribution')
#     plt.xticks(list(class_label_to_name.keys()), list(class_label_to_name.values()))
#     plt.tight_layout()
#
#     # Annotate the bars with the number of images
#     for class_label, count in class_counts_val.items():
#         plt.text(class_label, count + 5, str(count), ha='center', va='bottom')
#
#     # Save the validation set plot as an image file
#     val_plot_save_path = 'validation_class_distribution_plot.png'
#     plt.savefig(val_plot_save_path)
#
#     # Display the validation set plot
#     plt.show()
#
#     # Create a bar plot for the test set class distribution
#     plt.figure(figsize=(6, 5))
#     plt.bar(class_counts_test.keys(), class_counts_test.values())
#     plt.xlabel('Class Labels')
#     plt.ylabel('Number of Images')
#     plt.title('Test Set Class Distribution')
#     plt.xticks(list(class_label_to_name.keys()), list(class_label_to_name.values()))
#     plt.tight_layout()
#
#     # Annotate the bars with the number of images
#     for class_label, count in class_counts_test.items():
#         plt.text(class_label, count + 5, str(count), ha='center', va='bottom')
#
#     # Save the test set plot as an image file
#     test_plot_save_path = 'test_class_distribution_plot.png'
#     plt.savefig(test_plot_save_path)
#
#     # Display the test set plot
#     plt.show()
# except (ValueError, RuntimeError) as e:
#     print(e)
#


# Normalizing the data for the neural network
X_train = X_train / 255.0
X_test = X_test / 255.0
X_val = X_val/255.0

print(X_train[1].shape)

save_dir = 'Processed Data/Tuberculosis Kaggle Processed Data'
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
np.save(os.path.join(save_dir, 'Y_train.npy'), Y_train)
np.save(os.path.join(save_dir, 'Y_val.npy'), Y_val)
np.save(os.path.join(save_dir, 'Y_test.npy'), Y_test)
