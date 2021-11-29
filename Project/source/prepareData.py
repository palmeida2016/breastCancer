# https://www.kaggle.com/pratyushpatnaik/breast-cancer-detector-0-86-accuracy

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
import shutil


# Define Model Class
class Classifier():
	def __init__(self):
		self.input_path = r'input'
		self.dest_path = r'data'

		self.img_rows = 50
		self.img_cols = 50


	def loadPaths(self):
		print('Loading input paths')
		self.paths = [(self.input_path, x, i, img) for x in os.listdir(self.input_path) for i in os.listdir(os.path.join(self.input_path,x)) for img in os.listdir(os.path.join(self.input_path, x, i))]
		print(f'Found {len(self.paths)} images.')

	def saveImages(self):
		random.shuffle(self.paths)

		train, valid, test = np.split(self.paths, [int(.6 * len(self.paths)), int(.8 * len(self.paths))])

		for i in tqdm(train):
			dir_name, ids, cl, img = i
			shutil.copy(os.path.join(dir_name,ids,cl,img), os.path.join(self.dest_path, 'train', cl, img))

		for i in tqdm(valid):
			dir_name, ids, cl, img = i
			shutil.copy(os.path.join(dir_name,ids,cl,img), os.path.join(self.dest_path, 'valid', cl, img))
			
		for i in tqdm(test):
			dir_name, ids, cl, img = i
			shutil.copy(os.path.join(dir_name,ids,cl,img), os.path.join(self.dest_path, 'test', cl, img))


	def format(self):
		self.loadPaths()
		self.saveImages()

if __name__ == '__main__':
	f = Classifier()		
	f.format()