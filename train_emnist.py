import scipy.io
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Load EMNIST Letters MATLAB file
mat = scipy.io.loadmat('emnist-letters.mat')

# Extract images and labels
x_train = mat['dataset']['train'][0,0]['images'][0,0]  # (124800, 784)
y_train = mat['dataset']['train'][0,0]['labels'][0,0]  # (124800, 1)
x_test = mat['dataset']['test'][0,0]['images'][0,0]    # (20800, 784)
y_test = mat['dataset']['test'][0,0]['labels'][0,0]    # (20800, 1)

# Reshape images to 28x28
x_train = x_train.reshape(-1,28,28,1).astype('float32') / 255.0
x_test = x_test.reshape(-1,28,28,1).astype('float32') / 255.0

# Convert labels from 1-26 to 0-25
y_train = y_train - 1
y_test = y_test - 1

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=26)
y_test = to_categorical(y_test, num_classes=26)

# Build CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(26, activation='softmax')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('emnist_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)

# Train model
model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          epochs=30,
          batch_size=128,
          callbacks=[checkpoint, earlystop])

print("Training complete. Best model saved as emnist_model.h5")