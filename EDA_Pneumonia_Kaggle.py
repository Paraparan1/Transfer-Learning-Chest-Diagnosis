import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

data_dir = 'Pneumonia Kaggle Data/chest_xray'

class_names = ['TUBERCULOSIS', 'NORMAL']


processed_data = []

def preprocess_data():
    for class_name in class_names:
        for folder in ['train', 'test']:
            class_dir = os.path.join(data_dir, folder, class_name)
            class_num = class_names.index(class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img_arr = cv2.imread(img_path)
                try:
                    img_arr_resized = cv2.resize(img_arr, (224, 224))
                except Exception as e:
                    print(img_path)
                processed_data.append([img_arr_resized, class_num])

preprocess_data()

images = []
labels = []

for img, lab in processed_data:
    images.append(img)
    labels.append(lab)

example_images_dict = {class_name: [] for class_name in class_names}

# Traverse the labels to find unique ones and get their corresponding images
for i, label in enumerate(labels):
    class_name = class_names[label]
    if len(example_images_dict[class_name]) < 3:
        example_images_dict[class_name].append(images[i])

fig, axes = plt.subplots(2, 3, figsize=(15, 15))

for i, class_name in enumerate(class_names):
    for j, img in enumerate(example_images_dict[class_name]):
        axes[i, j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB
        axes[i, j].axis('off')
    # Setting a centered title for each row above the middle column
    axes[i, 1].set_title(class_name, y=1.15, fontsize=16)

plt.tight_layout()
plt.show()





