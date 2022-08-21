# import sys
# sys.path.append('..')
# from vlimit import vram_limit
# vram_limit()
import tensorflow as tf

 
from numba import cuda 
device = cuda.get_current_device()
device.reset()

# Load the Fashion MNIST dataset
fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

# Normalize the pixel values
training_images = training_images / 255.0
test_images = test_images / 255.0


#Callback implementation
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        '''
        Halts the training after reaching 60 percent accuracy

        Args:
          epoch (integer) - index of epoch (required but unused in the function definition below)
          logs (dict) - metric results from the training epoch
        '''

        # Check accuracy
        if(logs.get('acc') > 0.9):

          # Stop if threshold is met
            print("\nAccuracy is higher than 0.9 so cancelling training!")
            self.model.stop_training = True

# Instantiate class
callbacks = myCallback()

# Define the model
model = tf.keras.models.Sequential([
#     Add convolutions and max pooling
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Add the same layers as before
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    ])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

# Train the model with a callback
model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])