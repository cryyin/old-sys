# import the necessary packages
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import datasets,layers,optimizers,models,regularizers
from keras import backend as K
#K.set_image_dim_ordering('th')
import numpy as np

class vgg16:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1


		model.add(layers.Conv2D(64,(3,3),padding='same',input_shape=inputShape))
		model.add(layers.Activation('relu'))
		model.add(layers.BatchNormalization(axis=chanDim))
		model.add(layers.Dropout(0.3))

		model.add(layers.Conv2D(64,(3,3),padding='same'))
		model.add(layers.Activation('relu'))
		model.add(layers.BatchNormalization(axis=chanDim))

		model.add(layers.MaxPooling2D(pool_size=(2,2)))

		model.add(layers.Conv2D(128,(3,3),padding='same'))
		model.add(layers.Activation('relu'))
		model.add(layers.BatchNormalization(axis=chanDim))
		model.add(layers.Dropout(0.4))

		model.add(layers.Conv2D(128,(3,3),padding='same'))
		model.add(layers.Activation('relu'))
		model.add(layers.BatchNormalization(axis=chanDim))

		model.add(layers.MaxPooling2D(pool_size=(2,2)))

		model.add(layers.Conv2D(256,(3,3),padding='same'))
		model.add(layers.Activation('relu'))
		model.add(layers.BatchNormalization(axis=chanDim))
		model.add(layers.Dropout(0.4))

		model.add(layers.Conv2D(256,(3,3),padding='same'))
		model.add(layers.Activation('relu'))
		model.add(layers.BatchNormalization(axis=chanDim))
		model.add(layers.Dropout(0.4))

		model.add(layers.Conv2D(256,(3,3),padding='same'))
		model.add(layers.Activation('relu'))
		model.add(layers.BatchNormalization(axis=chanDim))

		model.add(layers.MaxPooling2D(pool_size=(2,2)))

		model.add(layers.Conv2D(512,(3,3),padding='same'))
		model.add(layers.Activation('relu'))
		model.add(layers.BatchNormalization(axis=chanDim))
		model.add(layers.Dropout(0.4))

		model.add(layers.Conv2D(512,(3,3),padding='same'))
		model.add(layers.Activation('relu'))
		model.add(layers.BatchNormalization(axis=chanDim))
		model.add(layers.Dropout(0.4))

		model.add(layers.Conv2D(512,(3,3),padding='same'))
		model.add(layers.Activation('relu'))
		model.add(layers.BatchNormalization(axis=chanDim))

		model.add(layers.MaxPooling2D(pool_size=(2,2)))

		model.add(layers.Conv2D(512,(3,3),padding='same'))
		model.add(layers.Activation('relu'))
		model.add(layers.BatchNormalization(axis=chanDim))
		model.add(layers.Dropout(0.4))

		model.add(layers.Conv2D(512,(3,3),padding='same',))
		model.add(layers.Activation('relu'))
		model.add(layers.BatchNormalization(axis=chanDim))
		model.add(layers.Dropout(0.4))

		model.add(layers.Conv2D(512,(3,3),padding='same'))
		model.add(layers.Activation('relu'))
		model.add(layers.BatchNormalization(axis=chanDim))

		model.add(layers.MaxPooling2D(pool_size=(2,2)))
		model.add(layers.Dropout(0.5))

		model.add(layers.Flatten())
		model.add(layers.Dense(512))
		model.add(layers.Activation('relu'))
		model.add(layers.BatchNormalization(axis=chanDim))

		model.add(layers.Dropout(0.5))
		model.add(layers.Dense(classes))
		model.add(layers.Activation('softmax'))


		model.build(input_shape=(None,32,32,3))

		model.summary()



		# return the constructed network architecture
		return model
