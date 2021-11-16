# https://www.kaggle.com/pratyushpatnaik/breast-cancer-detector-0-86-accuracy

# Import Keras Layers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Input
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import keras_preprocessing.image as IMAGE


#Import Train Test Split
from sklearn.model_selection import train_test_split

#Import Basic Libraries
import random
import numpy as np
import pandas as pd
import os
import itertools
from PIL import Image
import cv2
import glob
import time
from tqdm import tqdm


# Define Model Class
class Classifier(Sequential):
	def __init__(self, activation = 'relu'):
		super().__init__()

		self.input_path = r'input'
		self.activation = activation

		self.batch_size = 32
		self.epochs = 25
		self.nb_validation_samples = 55507
		self.nb_train_samples = 166516


		self.img_rows = 50
		self.img_cols = 50

		self.createArchitecture()

	def createArchitecture(self):
		self.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(50,50,3),activation='relu'))
		self.add(MaxPooling2D(pool_size=(2,2)))
		self.add(MaxPooling2D(pool_size=(2,2)))
		self.add(Dropout(0.25))
		self.add(Conv2D(filters=32,kernel_size=(4,4),activation='relu'))
		self.add(MaxPooling2D(pool_size=(2,2)))
		self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
		self.add(Dropout(0.25))
		self.add(Flatten())
		self.add(Dense(64,activation='relu'))
		self.add(Dense(1,activation='sigmoid'))


		# Compile
		self.compile(loss='categorical_crossentropy',
			optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
			metrics=['accuracy'])

		print(self.summary())

		self.initCallbacks()
		self.initDataGenerator()

	def initDataGenerator(self):
		train_data_dir = 'data/train'
		validation_data_dir = 'data/valid'

		train_datagen = ImageDataGenerator(
							rescale=1./255,
							rotation_range=30,
							horizontal_flip=True,
							fill_mode='nearest')

		validation_datagen = ImageDataGenerator(rescale=1./255)

		self.train_generator = train_datagen.flow_from_directory(
							train_data_dir,
							color_mode='rgb',
							target_size=(self.img_rows,self.img_cols),
							batch_size=self.batch_size,
							class_mode='binary',
							shuffle=True)

		self.validation_generator = validation_datagen.flow_from_directory(
									validation_data_dir,
									color_mode='rgb',
									target_size=(self.img_rows,self.img_cols),
									batch_size=self.batch_size,
									class_mode='binary',
									shuffle=True)

		print(self.train_generator.class_indices)

	def initCallbacks(self):
		checkpoint = ModelCheckpoint('weights.h5',
			monitor='val_loss',
			mode='min',
			save_best_only=True,
			verbose=1)

		earlystop = EarlyStopping(monitor='val_loss',
			min_delta=0,
			patience=3,
			verbose=1,
			restore_best_weights=True)

		reduce_lr = ReduceLROnPlateau(monitor='val_loss',
			factor=0.2,
			patience=3,
			verbose=1,
			min_delta=0.0001)

		self.callbacks = [earlystop,checkpoint,reduce_lr]


	def loadPaths(self):
		print('Loading input paths')
		self.paths = [os.path.join(self.input_path, x, i, img) for x in os.listdir(self.input_path) for i in os.listdir(os.path.join(self.input_path,x)) for img in os.listdir(os.path.join(self.input_path, x, i))]
		print(f'Found {len(self.paths)} images.')

	def loadImages(self):
		# Convert images to array
		print('Loading Images')
		data = [(IMAGE.img_to_array(IMAGE.load_img(path, target_size=(self.img_rows, self.img_cols))), int(path[-5])) for path in tqdm(self.paths)]
		print('Finished Loading Images')

		# Prepare for Test Split
		x,y = (np.stack([i[0] for i in data])/255 ,np.array([i[1] for i in data]))

		print(x,y)

		self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(x, y, random_state=0, test_size=0.3)


	def train(self):
		# self.loadPaths()

		# self.loadImages()

		self.history = self.fit(
			self.train_generator,
            steps_per_epoch=self.nb_train_samples//self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            validation_data=self.validation_generator,
            validation_steps=self.nb_validation_samples//self.batch_size)


if __name__ == '__main__':
	# import tensorflow as tf
	# with tf.device('/device:GPU:0'):
	# gpus = tf.config.experimental.list_physical_devices('GPU')
	# tf.config.experimental.set_memory_growth(gpus[0], True)
	c = Classifier()		
	c.train()