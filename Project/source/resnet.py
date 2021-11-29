
# Import Keras Layers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Layer, Concatenate, Dense, Dropout, Activation, Flatten, BatchNormalization, Input
from keras.layers import Conv2D, MaxPooling2D, AvgPool2D
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger
from keras.applications.vgg16 import preprocess_input
import keras_preprocessing.image as IMAGE


#Import Basic Libraries
import random
import os
from PIL import Image
from tqdm import tqdm


# Define Custom Layers
class MCDropout(Dropout):
	def call(self, inputs):
		return super(rate = 0.45).call(inputs, training=True)

class ResnetLayer(Layer):
	def __init__(self, filters, n_conv=4, kernel_size=3, strides=1, activation = 'relu', **kwargs):
		super().__init__(**kwargs)
		self.filters = filters
		self.n_conv = n_conv
		self.kernel_size = kernel_size
		self.strides = strides
		self.activation = activation


		self.layers = []

		for _ in range(n_conv):
			self.layers.append(Conv2D(
				filters = filters,
				kernel_size = kernel_size,
				strides = strides,
				padding = 'same'
				))

			self.layers.append(Activation(activation))

			self.layers.append(BatchNormalization())

	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'filters': self.filters,
			'n_conv': self.n_conv,
			'kernel_size': self.kernel_size,
			'strides': self.strides,
			'activation': self.activation,
		})
		return config


	def call(self, inputs, activation = 'relu'):
		output = inputs
		for layer in self.layers:
			output = layer(output)

		return relu(Concatenate()([output, inputs]))


class InceptionModule(Layer):
	def __init__(self, filter_list, **kwargs):
		super().__init__(**kwargs)
		self.filter_list = filter_list
		self.layers = []

		self.initStructure(filter_list)

	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'filter_list': self.filter_list
		})
		return config

	def initStructure(self,filter_list, activation = 'relu'):
		# Check if filter_list is big enough for all conditions
		assert len(filter_list) > 3

		# Construct Layers
		for i in range(len(filter_list)):
			if i == 0:
				self.layers.append((Conv2D(filters=filter_list[i],
					kernel_size = 1,
					strides = 1,
					padding = 'same'),

					Activation(activation)))
			elif i == len(filter_list) - 1:
				self.layers.append((MaxPooling2D(pool_size = 3,
					strides = 1,
					padding = 'same'),

					Activation(activation),

					Conv2D(filters=filter_list[i],
					kernel_size = 1,
					strides = 1,
					padding = 'same')))

			else:
				self.layers.append((Conv2D(filters=filter_list[i],
					kernel_size = 1,
					strides = 1,
					padding = 'same'),

					Activation(activation),

					Conv2D(filters=filter_list[i],
					kernel_size = 2*i-1,
					strides = 1,
					padding = 'same')))

	def call(self, inputs):
		outputs = []
		for layer in self.layers:
			# Single-Layered line
			if len(layer) == 1:
				outputs.append(layer(inputs))
				continue

			# Multi-Layered Line
			temp = inputs
			for module in layer:
				temp = module(temp)
			outputs.append(temp)

		return relu(Concatenate()(outputs))


# Define Model Class
class Classifier():
	def __init__(self, blocks = 4, filters = 32, activation = 'relu'):
		self.input_path = r'input'
		self.activation = activation

		self.batch_size = 32
		self.epochs = 100
		self.nb_validation_samples = 55507
		self.nb_train_samples = 166516

		# Input Dimensions
		self.img_rows = 50
		self.img_cols = 50
		self.channels = 3

		self.createArchitecture(blocks, filters)

	def createArchitecture(self, blocks, filters, rate = 0.5):
		# Construct Model

		# Inputs to Model
		inputs = Input(shape = (self.img_rows, self.img_cols, self.channels))

		# Block 1
		outputs = ResnetLayer(filters=filters)(inputs)
		outputs = AvgPool2D(pool_size=2,strides=2)(outputs)

		# Apply All Layers until 1x1
		for i in range(blocks):
			if i % 2 == 0:
				outputs = InceptionModule(filter_list = [int(filters / 4), filters, filters*2, int(filters/2)])(outputs)
			else:
				outputs = ResnetLayer(filters = filters)(outputs)
			
			outputs = AvgPool2D(pool_size = 2, strides = 2)(outputs)

		# 4096 Conv Layers
		outputs = MCDropout(rate)(outputs)
		outputs = Conv2D(filters = 4096, kernel_size = 1, strides = 1, padding = 'valid', activation = 'relu')(outputs)
		outputs = MCDropout(rate)(outputs)
		outputs = Conv2D(filters = 4096, kernel_size = 1, strides = 1, padding = 'valid', activation = 'relu')(outputs)
		outputs = Flatten()(outputs)

		# Reduce to categories
		outputs = Dense(units=2,activation='softmax')(outputs)

		# Create Model
		self.model = Model(inputs=inputs,outputs=outputs)

		# Compile
		self.model.compile(loss='categorical_crossentropy',
			optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
			metrics=['accuracy'])

		print(self.model.summary())

		self.initCallbacks()
		self.initDataGenerator()

	def initDataGenerator(self):
		train_data_dir = 'data/train'
		validation_data_dir = 'data/valid'
		test_data_dir = 'data/test'

		train_datagen = ImageDataGenerator(
							preprocessing_function=preprocess_input,
							horizontal_flip = True,
							vertical_flip = True,
							rescale=1./255)

		validation_datagen = ImageDataGenerator(
							preprocessing_function=preprocess_input,
							horizontal_flip = True,
							vertical_flip = True,
							rescale=1./255)

		test_datagen = ImageDataGenerator(
							preprocessing_function=preprocess_input,
							horizontal_flip = True,
							vertical_flip = True,
							rescale=1./255)

		self.train_generator = train_datagen.flow_from_directory(
							train_data_dir,
							color_mode='rgb',
							target_size=(self.img_rows,self.img_cols),
							batch_size=self.batch_size,
							shuffle=True)

		self.validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='rgb',
							target_size=(self.img_rows,self.img_cols),
							batch_size=self.batch_size,
							shuffle=True)

		self.test_generator = test_datagen.flow_from_directory(
							test_data_dir,
							color_mode='rgb',
							target_size=(self.img_rows,self.img_cols),
							batch_size=self.batch_size,
							shuffle=True)	

	def initCallbacks(self):
		checkpoint = ModelCheckpoint('skin_disease.h5',
			monitor='val_loss',
			mode='min',
			save_best_only=True,
			verbose=1)

		earlystop = EarlyStopping(monitor='val_loss',
			min_delta=0,
			patience=10,
			verbose=1,
			restore_best_weights=True)

		reduce_lr = ReduceLROnPlateau(monitor='val_loss',
			factor=0.2,
			patience=3,
			verbose=1,
			min_delta=0.0001)

		tensor_board = TensorBoard(log_dir="logs",
			histogram_freq=0,
			write_graph=True,
			write_images=False,
			update_freq="epoch",
			profile_batch=2,
			embeddings_freq=0,
			embeddings_metadata=None
			)

		self.callbacks = [checkpoint, earlystop, reduce_lr, tensor_board]

	def train(self):
		self.history = self.model.fit(
			self.train_generator,
			steps_per_epoch=self.nb_train_samples//self.batch_size,
			epochs=self.epochs,
			callbacks=self.callbacks,
			validation_data=self.validation_generator,
			validation_steps=self.nb_validation_samples//self.batch_size)

	def test(self):
		loss = []
		accs = []
		for _ in range(10):
			score = self.model.evaluate(self.test_generator)
			loss.append(score[0])
			accs.append(score[1])


		import statistics
		print('Test loss:', statistics.mean(loss)) 
		print('Test accuracy:', statistics.mean(accs))

	def loadWeights(self):
		self.model.load_weights('skin_disease.h5')

	def gradCam(self):
		# https://github.com/wawaku/grad-cam-keras/blob/master/grad-cam.py
		pass

	def getSampleDataset(self, n):
		pass

	def predict(self):
		pass
		

if __name__ == '__main__':
	c = Classifier()
	# c.loadWeights()
	c.train()		
	# c.test()