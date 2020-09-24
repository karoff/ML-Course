# Scientific and vector computation for python
import numpy as np
np.random.seed(42)  # Set the global random seed to make reproducible experiments (scikit-learn also use this)

# Deep learning framework
from keras.datasets import mnist  # Load MNIST dataset
from keras.models import Sequential  # Create models sequentially
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D  # Relevant layers
from keras.optimizers import Adam  # Optimizer for gradient descent
from keras.backend import clear_session  # Delete previous models
from keras.utils import to_categorical

import matplotlib.pyplot as plt  

# Load the data (already split between train and test sets)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Input image dimensions and number of classes
img_rows, img_cols = 28, 28
num_classes = 10

# Pre-process data
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Set hyperparameters
batch_size = 128
epochs = 5

# Define the input shape
input_shape = (img_rows, img_cols, 1)

# Now create the model
# See inspiration here: https://keras.io/getting-started/sequential-model-guide/
# ====================== YOUR CODE HERE =======================

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# =============================================================

# Print a summary of the defined model
model.summary()

# Compile the model using categorical crossentropy as loss function and the Adam optimizer for gradient descent
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['acc'])

# Train the model and save the loss and accuracy in history
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test))

# Plot training & validation accuracy values
plt.figure(figsize=(8, 6))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('acc2.png')

# Plot training & validation loss values
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('loss2.png')

plt.figure(figsize=(8, 8))
for i in range(32):
    plt.subplot(6, 6, i+1)
    plt.imshow(model.layers[0].get_weights()[0][:, :, :, i].squeeze(), cmap='gray')
    plt.axis('off')
    plt.title("Filter no.: " + str(i + 1))
plt.tight_layout()
plt.savefig('filters.png')

