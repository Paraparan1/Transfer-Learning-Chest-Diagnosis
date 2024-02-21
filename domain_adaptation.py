import os
import random
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from keras.models import load_model


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

random.seed(29)

# Set random seed for NumPy
np.random.seed(29)

# Set random seed for TensorFlow
tf.random.set_seed(29)

tf.debugging.set_log_device_placement(True)

gpus = tf.config.list_physical_devices('GPU')

for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Invalid device or cannot modify virtual devices once initialized.
        print(e)


device_names = ['/gpu:' + str(i) for i in range(len(tf.config.list_physical_devices('GPU')))]
strategy = tf.distribute.MirroredStrategy(devices=device_names)


#class_names = ['Tuberculosis', 'Normal']

class_names = ['Pneumonia','Normal']


load_dir = 'dataset 1'
X_train = np.load(os.path.join(load_dir, 'X_train.npy'))
X_val = np.load(os.path.join(load_dir, 'X_val.npy'))
X_test = np.load(os.path.join(load_dir, 'X_test.npy'))
Y_train = np.load(os.path.join(load_dir, 'Y_train.npy'))
Y_val = np.load(os.path.join(load_dir, 'Y_val.npy'))
Y_test = np.load(os.path.join(load_dir, 'Y_test.npy'))

X_val = X_val/255.0

model = load_model('inceptionv3_Tuberculosis_Kaggle_Model.h5')

# Evaluate the model on the test set and print performance metrics
loss, accuracy = model.evaluate(X_test, Y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
conf_matrix = confusion_matrix(Y_test, Y_pred_classes)

# Plot Confusion Matrix
plt.figure(figsize=(15, 15))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
disp.ax_.set_title('Confusion Matrix')
disp.ax_.set_xlabel('Predicted Label')
disp.ax_.set_ylabel('True Label')
plt.tight_layout()
plt.savefig('DomainAdapt_confusion_matrix_inceptiond2_d1.png')

precision = precision_score(Y_test, Y_pred_classes, average='weighted')
recall = recall_score(Y_test, Y_pred_classes, average='weighted')
f1 = f1_score(Y_test, Y_pred_classes, average='weighted')

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
