# Scientific and vector computation for python
import numpy as np
np.random.seed(42)  # Set the global random seed to make reproducible experiments (scikit-learn also use this)

# Used for manipulating directory paths
import os

# Library to handle images
from PIL import Image, ImageOps

# Used to delete directories
import shutil

# Unzip files
import zipfile

# Split a directory of images into two directories containing train and test images
import splitfolders

# Deep learning framework
from keras.models import Sequential  # Create models sequentially
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Conv2D, MaxPooling2D  # Relevant layers
from keras.optimizers import Adam  # Optimizer for gradient descent
from keras.backend import clear_session  # Delete previous models
from keras.preprocessing.image import ImageDataGenerator  # To feed the model with images during training

# Ignore warning for corrupt EXIF data in the images
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

# Plotting library
import matplotlib.pyplot as plt  

# Check if data has already been processed
if not os.path.exists(os.path.join('Data', 'kagglecatsanddogs_3367a', 'processed')):
    print("Processing data.")
    
    # Delete any previous 'kagglecatsanddogs' datasets, if they should be present
    try:
        shutil.rmtree(os.path.join('Data', 'kagglecatsanddogs_3367a'))
    except:
        pass

    # Unzip images
    with zipfile.ZipFile("kagglecatsanddogs_3367a.zip","r") as zip_ref:
    # zip_ref.extractall(os.path.join('Data', 'kagglecatsanddogs', 'raw'))
        zip_ref.extractall(os.path.join('Data'))

    # Remove two corrupt images
    os.remove(os.path.join('Data', 'kagglecatsanddogs_3367a', 'PetImages', 'Cat', '666.jpg'))
    os.remove(os.path.join('Data', 'kagglecatsanddogs_3367a', 'PetImages', 'Dog', '11702.jpg'))

    # Split dataset into train and test set
    splitfolders.ratio(os.path.join('Data', 'kagglecatsanddogs_3367a', 'PetImages'), 
                        output=os.path.join('Data', 'kagglecatsanddogs_3367a', 'processed'), 
                        seed=42, 
                        ratio=(.7, 0, .3))
else:
    print("It seems like the data has already been processed.")
    
# Dataset details
train_samples = 17498
test_samples = 7500

# Hyperparameters
img_width, img_height = 128, 128  # Size you want to rescale images to
batch_size = 32
epochs = 5  # In this exercise, 50 epochs corresponds to training on the entire dataset once

# Use data augmentation during training
datagen = ImageDataGenerator(rescale=1./255)  # Re-scale images to pixel values between 0 and 1


train_generator = datagen.flow_from_directory('Data/kagglecatsanddogs_3367a/processed/train',
                                              target_size=(img_width, img_height),
                                              batch_size=batch_size,
                                              class_mode='binary')

test_generator = datagen.flow_from_directory('Data/kagglecatsanddogs_3367a/processed/test',
                                             target_size=(img_width, img_height),
                                             batch_size=batch_size,
                                             class_mode='binary')

# Define model
# ====================== YOUR CODE HERE =======================



# =============================================================

model.compile(optimizer=Adam(lr=1e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

model.summary()

history = model.fit_generator(train_generator,
                              steps_per_epoch=train_samples/batch_size/50,
                              epochs=epochs,
                              validation_data=test_generator,
                              validation_steps=test_samples/batch_size/50)

model.save("model_Custom.h5")

# Plot training & validation accuracy values
plt.figure(figsize=(8, 6))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('acc3.png')

# Plot training & validation loss values
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('loss3.png')



