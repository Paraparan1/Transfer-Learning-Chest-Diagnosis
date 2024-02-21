import os
import random
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping

random.seed(29)

# Set random seed for NumPy
np.random.seed(29)

# Set random seed for TensorFlow
tf.random.set_seed(29)

class_names = ['NORMAL', 'PNEUMONIA']

load_dir = 'dataset 1'
X_train = np.load(os.path.join(load_dir, 'X_train.npy'))
X_val = np.load(os.path.join(load_dir, 'X_val.npy'))
X_test = np.load(os.path.join(load_dir, 'X_test.npy'))
Y_train = np.load(os.path.join(load_dir, 'Y_train.npy'))
Y_val = np.load(os.path.join(load_dir, 'Y_val.npy'))
Y_test = np.load(os.path.join(load_dir, 'Y_test.npy'))


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size = 32
epochs = 100

early_stopping = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')

history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=(X_val, Y_val), callbacks=[early_stopping])
model.save("CNN_Tuberculosis_Kaggle.h5")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('cnn_trainloss_Tuberculosis.png')

# Evaluate the model on the test set and print performance metrics
loss, accuracy = model.evaluate(X_test, Y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
conf_matrix = confusion_matrix(Y_test, Y_pred_classes)

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
disp.ax_.set_title('Confusion Matrix')
disp.ax_.set_xlabel('Predicted Label')
disp.ax_.set_ylabel('True Label')
plt.savefig('CNN_confusion_matrix_Tuberculosis.png')

precision = precision_score(Y_test, Y_pred_classes, average='weighted')
recall = recall_score(Y_test, Y_pred_classes, average='weighted')
f1 = f1_score(Y_test, Y_pred_classes, average='weighted')

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

