import os
import random
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.keras.callbacks import EarlyStopping

random.seed(29)

# Set random seed for NumPy
np.random.seed(29)

# Set random seed for TensorFlow
tf.random.set_seed(29)

#Class names
class_names = ['NORMAL', 'PNEUMONIA']

#Loading Datasets
load_dir = 'dataset 1'
X_train = np.load(os.path.join(load_dir, 'X_train.npy'))
X_val = np.load(os.path.join(load_dir, 'X_val.npy'))
X_test = np.load(os.path.join(load_dir, 'X_test.npy'))
Y_train = np.load(os.path.join(load_dir, 'Y_train.npy'))
Y_val = np.load(os.path.join(load_dir, 'Y_val.npy'))
Y_test = np.load(os.path.join(load_dir, 'Y_test.npy'))

# Resnet Model
base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))

model = Sequential()

model.add(base_model)

model.add(GlobalAveragePooling2D())

model.add(Dense(256, activation='relu'))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

batch_size = 32
epochs = 100

early_stopping = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')

# Train the model with early stopping
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=(X_val, Y_val), callbacks=[early_stopping])


model.save("Resnet_pneumonia_Kaggle_Model_nw.h5")

#Training and Validation loss plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('resNet_loss_curve_pneumonia_nw.png')

# Evaluate the model on the test set and print performance metrics
loss, accuracy = model.evaluate(X_test, Y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

#Store predicted values in Y pred for confusion matrix.
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
conf_matrix = confusion_matrix(Y_test, Y_pred_classes)

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
disp.ax_.set_title('Confusion Matrix')
disp.ax_.set_xlabel('Predicted Label')
disp.ax_.set_ylabel('True Label')
plt.savefig('resnet_confusion_matrix_pneumonia_nw.png')

#Calculate precison, recall, f1 score
precision = precision_score(Y_test, Y_pred_classes, average='weighted')
recall = recall_score(Y_test, Y_pred_classes, average='weighted')
f1 = f1_score(Y_test, Y_pred_classes, average='weighted')


print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

