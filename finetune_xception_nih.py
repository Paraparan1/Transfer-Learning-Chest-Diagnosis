import os
import pickle
import random
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import ReduceLROnPlateau

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
partitions = [partition_1_generator, partition_2_generator, partition_3_generator]
all_histories = []

initial_epochs = 10
fine_tune_epochs = 30
total_epochs = initial_epochs + fine_tune_epochs

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='fine_tuned_checkpoint.h5',
                                                         save_best_only=True,
                                                         monitor='val_loss',
                                                         mode='min',
                                                         verbose=1)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=10,
                                                           restore_best_weights=True,
                                                           monitor='val_loss',
                                                           verbose=1)

reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.1,
                                         patience=5,
                                         verbose=1,
                                         mode='min',
                                         min_lr=0.00001)


for i, partition_generator in enumerate(partitions):
    print(f"Training on Partition {i + 1}...")

    # Load the pretrained model
    model = tf.keras.models.load_model('xception_NIH_Model.h5')

    # Freeze the base layers
    for layer in model.layers[:-3]:  # The last 3 layers are the GAP, Dense(256), and Dense(15)
        layer.trainable = False

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Initial training with base layers frozen
    history = model.fit(partition_generator,
                        epochs=initial_epochs,
                        validation_data=val_generator,
                        callbacks=[checkpoint_callback, early_stopping_callback,reduce_lr_on_plateau])

    # Unfreeze some of the top layers of the base model
    for layer in model.layers[-20:]:
        layer.trainable = True

    # Fine-tuning
    history_fine = model.fit(partition_generator,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_data=val_generator,
                             callbacks=[checkpoint_callback, early_stopping_callback,reduce_lr_on_plateau])

    # Store histories
    all_histories.append((history, history_fine))

with open('history_finetune', 'wb') as file:
    pickle.dump(all_histories, file)

# Evaluate the final model
loss, accuracy = model.evaluate(test_generator)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Plotting the training and validation loss and accuracy
plt.figure(figsize=(10, 6))
for i, (history, history_fine) in enumerate(all_histories):
    plt.plot(history.history['loss'], label=f'Partition {i + 1} - Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss for All Partitions')
plt.tight_layout()
plt.savefig('ft_training_loss_plot.png')

plt.figure(figsize=(10, 6))
for i, (history, history_fine) in enumerate(all_histories):
    plt.plot(history.history['val_loss'], label=f'Partition {i + 1} - Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Validation Loss for All Partitions')
plt.tight_layout()
plt.savefig('ft_validation_loss_plot.png')

model.save('fine_tuned_xception_NIH_Model.h5')

model = load_model('fine_tuned_xception_NIH_Model.h5')

Y_pred = model.predict(test_generator, steps=len(test_generator))
Y_pred_classes = np.argmax(Y_pred, axis=1)

precision = precision_score(Y_test, Y_pred_classes, average='weighted')
recall = recall_score(Y_test, Y_pred_classes, average='weighted')
f1 = f1_score(Y_test, Y_pred_classes, average='weighted')

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)