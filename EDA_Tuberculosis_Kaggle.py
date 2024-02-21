import os
import cv2
from matplotlib import pyplot as plt

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

