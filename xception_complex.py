import os
import random
import numpy as np
import datetime
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import pickle


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

class_names = ['No Finding', 'Infiltration', 'Atelectasis', 'Effusion', 'Nodule', 'Pneumothorax', 'Mass',
               'Effusion|Infiltration','Atelectasis|Infiltration', 'Consolidation', 'Atelectasis|Effusion',
               'Pleural_Thickening', 'Cardiomegaly','Emphysema', 'Infiltration|Nodule']



load_dir = 'dataset 3'
X_train = np.load(os.path.join(load_dir, 'X_train.npy'))
X_val = np.load(os.path.join(load_dir, 'X_val.npy'))
X_test = np.load(os.path.join(load_dir, 'X_test.npy'))
Y_train = np.load(os.path.join(load_dir, 'Y_train.npy'))
Y_val = np.load(os.path.join(load_dir, 'Y_val.npy'))
Y_test = np.load(os.path.join(load_dir, 'Y_test.npy'))

# Data preprocessing parameters
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

batch_size_gen = 16
total_samples = len(X_train)
partition_size = total_samples // 3

# Partition the data
partition_1_X = X_train[:partition_size]
partition_1_Y = Y_train[:partition_size]

partition_2_X = X_train[partition_size:2 * partition_size]
partition_2_Y = Y_train[partition_size:2 * partition_size]

partition_3_X = X_train[2 * partition_size:]
partition_3_Y = Y_train[2 * partition_size:]

# Create data generators for each partition
partition_1_generator = datagen.flow(
    partition_1_X, partition_1_Y,
    batch_size=batch_size_gen,
    shuffle=True
)


partition_2_generator = datagen.flow(
    partition_2_X, partition_2_Y,
    batch_size=batch_size_gen,
    shuffle=True
)

partition_3_generator = datagen.flow(
    partition_3_X, partition_3_Y,
    batch_size=batch_size_gen,
    shuffle=True
)

# Create a generator for validation data
val_generator = datagen.flow(
    X_val, Y_val,
    batch_size=batch_size_gen,
    shuffle=False
)


# Create a generator for testing data
test_generator = datagen.flow(
    X_test, Y_test,
    batch_size=batch_size_gen,
    shuffle=False
)

start_time = datetime.datetime.now()
# Drop out and l2 regularization
dropout_rate = 0.5
l2_regularization = 0.01

with strategy.scope():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output

    x = SpatialDropout2D(0.25)(x)

    x = GlobalAveragePooling2D()(x)

    # Add a fully connected layer with dropout and L2 regularization
    x = Flatten()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(l2_regularization))(x)
    x = Dropout(dropout_rate)(x)

    # Add another fully connected layer with L2 regularization
    x = Dense(256, activation='relu', kernel_regularizer=l2(l2_regularization))(x)
    x = Dropout(0.3)(x)

    x = Dense(128, activation='relu', kernel_regularizer=l2(l2_regularization))(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(64, activation='relu', kernel_regularizer=l2(l2_regularization))(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(32, activation='relu', kernel_regularizer=l2(l2_regularization))(x)
    x = Dropout(dropout_rate)(x)

    # Output layer
    output = Dense(15, activation='softmax')(x)

    # Create the model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Print model summary
    model.summary()

# Define checkpoint callback
checkpoint_callback = ModelCheckpoint(filepath='complex_xception_checkpoint_nih.h5',
                                      save_best_only=True,
                                      save_weights_only=True,
                                      monitor='val_loss',
                                      mode='min',
                                      verbose=1)

# Define early stopping callback
early_stopping_callback = EarlyStopping(patience=10,
                                        restore_best_weights=True,
                                        monitor='val_loss',
                                        verbose=1)

# Train and save history for each partition
all_history = []

for i, partition_generator in enumerate([partition_1_generator, partition_2_generator, partition_3_generator]):
    weights_filepath = 'complex_xception_model_checkpoint_nih.h5'
    if os.path.exists(weights_filepath) and i > 0:
            model.load_weights(weights_filepath)
    steps_per_epoch = partition_size // batch_size_gen

    history = model.fit(partition_generator, epochs=100,
                        validation_data=val_generator,
                        callbacks=[checkpoint_callback, early_stopping_callback],
                        steps_per_epoch=steps_per_epoch)

    # Save history to disk
    all_history.append(history.history)

end_time = datetime.datetime.now()

model.save('complex_xception_NIH_Model.h5')

total_time = end_time - start_time

print(f"Total time taken to train: {total_time}")

# Save all history
with open('complex_xception_NIH_all_history.pkl', 'wb') as file:
    pickle.dump(all_history, file)

plt.figure()
for idx, history in enumerate(all_history):
    plt.plot(history['loss'], label=f'Partition {idx + 1}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('complex_xception_loss_curve_nih.png')

model = load_model('complex_xception_NIH_Model.h5')

# Evaluate the model using the test generator
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Generate predictions for the test set using the test generator
Y_pred = model.predict(test_generator, steps=len(test_generator))

# Convert predicted probabilities to predicted classes
Y_pred_classes = np.argmax(Y_pred, axis=1)

plt.figure(figsize=(15, 15))
conf_matrix = confusion_matrix(Y_test, Y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('complex_xception_confusion_matrix_nih.png')

precision = precision_score(Y_test, Y_pred_classes, average='weighted')
recall = recall_score(Y_test, Y_pred_classes, average='weighted')
f1 = f1_score(Y_test, Y_pred_classes, average='weighted')

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
