# Import all required keras libraries
from keras.datasets import mnist 
from keras.utils import to_categorical
from keras import Sequential
from keras import layers

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

# define CNN model here in this function and then later use this function to create your model
def digit_recognition_cnn():
	# create CNN model with Conv + ReLU + Flatten + Dense layers
  model = Sequential()
  # Conv Input
  model.add(layers.Conv2D(filters=30, kernel_size=(5, 5), activation='relu', input_shape=[28, 28, 1]))
  model.add(layers.MaxPool2D(pool_size=2, strides=2))
  # Conv
  model.add(layers.Conv2D(filters=15, kernel_size=(3, 3), activation='relu',))
  model.add(layers.MaxPool2D(pool_size=2, strides=2))
  # Flatten
  model.add(layers.Flatten())
  # Dense
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.Dense(50, activation='relu'))
  # Dense Output
  model.add(layers.Dense(10, activation='softmax'))
	# Compile model
  model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['accuracy'])
  
  return model

# build your model
X_train, X_test, y_train, y_test = load_dataset()
model = digit_recognition_cnn()

# Train model and see the result
model.fit(X_train, y_train, epochs=15, batch_size=150)

# Evaluate model
model.evaluate(X_test, y_test)

# Save model
model.save('digitRecognizer.h5')