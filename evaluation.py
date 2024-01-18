# Import all required keras libraries
from keras.models import load_model # This is used to load saved model
from keras.datasets import mnist # This is used to load mnist dataset later
from keras.utils import to_categorical # This will be used to convert your test image to a categorical class (digit from 0 to 9)

# load required keras libraries
from keras.preprocessing.image import load_img # This is required to load the image
from keras.preprocessing.image import img_to_array # This is required to convert the image to array

# Load and return training and test datasets
def load_dataset():
	# Load dataset via imported keras library
  (X_train, y_train), (X_test, y_test) = mnist.load_data()
	# reshape for X train and test vars
  X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
  X_train.shape

  X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
  X_test.shape

	# normalize inputs from 0-255 to 0-1
  X_train = X_train / 255
  X_test = X_test / 255

	# Convert y_train and y_test to categorical classes
  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)

  return X_train, X_test, y_train, y_test

# Load saved model
loaded_model = load_model('/content/drive/MyDrive/digitRecognizer.h5')

# Evaluate model
X_train, X_test, y_train, y_test = load_dataset()
loaded_model.evaluate(X_test, y_test)

# load and normalize new image
def load_new_image(path):
  # load new image
  newImage = load_img(path, grayscale=True, target_size=(28, 28))
  # Convert image to array
  newImage = img_to_array(newImage)
  # reshape into a single sample with 1 channel
  newImage = newImage.reshape(1, 28, 28, 1).astype('float32')
  # normalize image data
  newImage = newImage / 255
  
  return newImage

# load a new image and predict its class
def test_model_performance():
  img = load_new_image('/content/drive/MyDrive/sample_images/digit6.png')
  # predict the class
  imageClass = loaded_model.predict(img)
  # Print prediction result
  print(np.argmax(imageClass[0]))

# Test model performance
import numpy as np
test_model_performance()